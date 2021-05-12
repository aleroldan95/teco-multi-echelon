import os
import datetime as dt
import pandas as pd
import random
import time
from math import floor
from itertools import product
import plotly.express as px
import numpy as np
from data_loader_class import DataClass
from functions import create_query, postgres_connection

import ray
import ray.rllib.agents.impala as impala
from reinforcement_learning.redistribution_env import SimulationEnvironment
from tqdm import tqdm


class ModelVSHistory:

    heuristics = ["Action1", "Action2", "Action3", "Action4",
                  'Random', 'No movements', 'Action7', 'Action10', 'Action11', 'Action12',
                  'Action13', 'Neural Net']

    heuristic_mapper = {'Action1': 0,
                        'Action2': 1,
                        'Action3': 2,
                        'No movements': 3,
                        'Action7': 4,
                        'Action13': 5,
                        'Action14': 6,
                        'Action15': 7,
                        'Random': 8}

    def __init__(self, env_config, chosen_heuristics=None, geus_ids=None):
        self.env = SimulationEnvironment(env_config=env_config)

        self.trainer = None

        self.reset_data()

        self.geus_ids = geus_ids

        self.stock = {}
        self.on_time = {}
        self.total = {}

        self.clusters = DataClass.clusters
        self.domains = self.env.data_class.domain_list
        self.procurement_types = DataClass.procurement_types

        if chosen_heuristics:
            self.chosen_heuristics = chosen_heuristics
        else:
            self.chosen_heuristics = self.heuristics

        if 'Neural Net' in self.chosen_heuristics:
            ray.init(local_mode=True, include_dashboard=False)
            config = impala.impala.DEFAULT_CONFIG.copy()
            config["env"] = SimulationEnvironment
            if env_config:
                config["env_config"] = env_config
            config["num_workers"] = 1
            config["sample_async"] = False
            config["num_gpus"] = 0
            # config['model']["fcnet_hiddens"] = [32, 32],
            config["rollout_fragment_length"] = 48
            config["train_batch_size"] = config["rollout_fragment_length"] * 5
            self.trainer = impala.impala.ImpalaTrainer(config)
            trainer_path = os.path.join(os.path.dirname(__file__),
                                        "models/IMPALA/IMPALA_SimulationEnvironment_0_2020-10-15_21-22-39ocpvemq4/checkpoint_5000/checkpoint-5000")
            self.trainer.restore(trainer_path)
            self.net_chosen_actions = {heur: 0 for heur in self.heuristic_mapper.values()}

    def run(self, slice=None, timing=False, clusters=None, domains=None, procurement_types=None):
        if clusters:
            self.clusters = clusters
        if domains:
            self.domains = domains
        if procurement_types:
            self.procurement_types = procurement_types
        if timing:
            times = {heuristic: 0 for heuristic in self.chosen_heuristics}
        if self.geus_ids:
            geus = list(filter(lambda x: x.id in self.geus_ids,
                               self.env.data_class.geus_by_id.values()))
            random.shuffle(geus)
        else:
            geus = list(filter(lambda x: (x.cluster in self.clusters) & (x.procurement_type in self.procurement_types) &
                                         (x.domain in self.domains),
                               self.env.data_class.geus_by_id.values()))
            if not slice:
                slice = len(geus)
            else:
                slice = min(slice, len(geus))
            random.shuffle(geus)
            geus = geus[:slice]
            self.geus_ids = [geu.id for geu in geus]

        self.stock = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.on_time = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.total = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.mlen = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.mtime = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.mttime = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.mov_cost = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}
        self.compras = {geu.id: {key: {} for key in self.chosen_heuristics} for geu in geus}

        for geu in tqdm(geus):
            for heuristic in self.chosen_heuristics:
                if timing:
                    tic = time.time()
                self._simulate(geu, heuristic)
                if timing:
                    times[heuristic] += time.time() - tic

        self.stock_data = pd.DataFrame([(geu.id, heuristic, fecha, cram, stocks[cram])
                                        for geu in geus
                                        for heuristic in self.chosen_heuristics
                                        for fecha, stocks in self.stock[geu.id][heuristic].items()
                                        for cram in range(self.env.num_wh)],
                                       columns=["geu", "heuristica", "fecha", "cram", "stock"])

        self.service_data = pd.DataFrame([(geu.id, heuristic, fecha, cram,
                                           self.on_time[geu.id][heuristic][fecha][cram],
                                           self.total[geu.id][heuristic][fecha][cram])
                                          for geu in geus
                                          for heuristic in self.chosen_heuristics
                                          for fecha in self.on_time[geu.id][heuristic].keys()
                                          for cram in range(self.env.num_wh)],
                                         columns=["geu", "heuristica", "fecha", "cram", "on_time", "total"])

        self.travel_data = pd.DataFrame([(geu.id, heuristic, fecha,
                                          self.mlen[geu.id][heuristic][fecha],
                                          self.mtime[geu.id][heuristic][fecha],
                                          self.mttime[geu.id][heuristic][fecha],
                                          self.mov_cost[geu.id][heuristic][fecha])
                                          for geu in geus
                                          for heuristic in self.chosen_heuristics
                                          for fecha in self.mlen[geu.id][heuristic].keys()],
                                          columns=["geu", "heuristica", "fecha", "mlen", "mtime", "mttime", 'cost'])

        self.compras_data = pd.DataFrame([(geu.id, heuristic, fecha,
                                          self.compras[geu.id][heuristic][fecha])
                                          for geu in geus
                                          for heuristic in self.chosen_heuristics
                                          for fecha in self.compras[geu.id][heuristic].keys()],
                                          columns=["geu", "heuristica", "fecha", "compras"])

        self.entradas_data = self.process_compras(self.compras_data)

        if timing:
            times = {key: value / len(geus) for key, value in times.items()}
            print(times)

    def reset_data(self):
        self.stock_data = pd.DataFrame(columns=["geu", "cram", "fecha", "stock"])
        self.service_data = pd.DataFrame(columns=["geu", "cram", "fecha", "on_time", "total"])
        self.travel_data = pd.DataFrame(columns=["geu", "fecha", "mlen", "mtime", "mttime", 'cost'])
        self.compras_data = pd.DataFrame(columns=["geu", "fecha", "compras"])
        self.entradas_data = pd.DataFrame(columns=["geu", "fecha", "compras", "reparaciones", "desmontes", "spm"])
        self.geus_ids = []

    def set_multipliers(self, multi):
        self.env.multiplicador = multi

    def _simulate(self, geu, heuristic):
        self.env.geu = geu
        obs = self.env.reset()
        while True:
            if heuristic != 'Neural Net':
                action = self.heuristic_mapper[heuristic]
            else:
                action = self.trainer.compute_action(obs)
                self.net_chosen_actions[action] += 1

            obs, _, done, info = self.env.step(action)
            #self.env.render()
            if done:
                break

        if info:
            self.stock[geu.id][heuristic] = info["Stock"]
            self.on_time[geu.id][heuristic] = info["On_time"]
            self.total[geu.id][heuristic] = info["Total"]
            self.mlen[geu.id][heuristic] = info["Mlen"]
            self.mtime[geu.id][heuristic] = info["Mtime"]
            self.mttime[geu.id][heuristic] = info["Mttime"]
            self.mov_cost[geu.id][heuristic] = info["mov_cost"]
            self.compras[geu.id][heuristic] = info["Compras"]

    def plot(self):
        subset = self.stock_data.groupby(["fecha", "heuristica"])["stock"].sum().reset_index()
        subset_compras = self.compras_data.groupby(["fecha", "heuristica"])["compras"].sum().reset_index()
        subset_compras[["compras_cum"]] = subset_compras.groupby('heuristica')['compras'].transform(pd.Series.cumsum)
        subset_tickets = self.service_data.groupby(["fecha", "heuristica"])[["on_time", "total"]].sum().reset_index()
        subset_tickets[['on_time', "total"]] = subset_tickets.groupby('heuristica')[['on_time', "total"]].transform(pd.Series.cumsum)
        subset_tickets["service_level"] = subset_tickets["on_time"] / subset_tickets["total"]
        subset_travel = self.travel_data.copy().sort_values("fecha")
        subset_travel[['mlen', "mtime", "mttime", 'cost']] = subset_travel.groupby('heuristica')[['mlen', "mtime", "mttime", 'cost']].transform(pd.Series.cumsum)

        fig1 = px.line(subset, x="fecha", y="stock", color='heuristica', title="Stock", height=600)
        fig2 = px.line(subset_tickets, x="fecha", y="service_level", title="Nivel de Servicio", color='heuristica', height=600)
        fig3 = px.line(subset_tickets, x="fecha", y="on_time", color='heuristica', title="Tickets Completados", height=600)
        fig4 = px.line(subset_tickets, x="fecha", y="total", color='heuristica', title="Tickets Totales", height=600)
        fig5 = px.line(subset_travel, x="fecha", y="mlen", color="heuristica", title="Cantidad de viajes", height=600)
        fig6 = px.line(subset_travel, x="fecha", y="mtime", color="heuristica", title="Tiempos de viaje x movimiento", height=600)
        fig7 = px.line(subset_travel, x="fecha", y="mttime", color="heuristica", title="Tiempos de viaje x material", height=600)
        fig8 = px.line(subset_travel, x="fecha", y="cost", color="heuristica", title="Costo de viaje", height=600)
        fig9 = px.line(subset_compras, x="fecha", y="compras_cum", color="heuristica", title="Compras", height=600)
        #fig5 = px.bar(subset5, x="fecha", y="stock", color="cram", title="Stock historico por cram",
        #              color_discrete_sequence=px.colors.qualitative.Vivid, height=600)

        figures = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]

        if 'Neural Net' in self.chosen_heuristics:
            valores = {}
            for action in self.heuristic_mapper:
                valores[action] = self.net_chosen_actions[self.heuristic_mapper[action]]
            amounts = pd.Series(valores).reset_index().rename({'index': 'Action', 0: 'Usos'}, axis=1)
            fig_actions = px.bar(amounts.sort_values('Usos', ascending=False), x='Action', y='Usos')
            figures.append(fig_actions)

        # Show plot
        self.figures_to_html(figures)

    def return_service_level(self):
        if self.service_data["total"].sum() == 0:
            return 1
        return round(self.service_data["on_time"].sum() / self.service_data["total"].sum() * 100,2)

    def figures_to_html(self, figs, filename=os.path.join(os.path.dirname(__file__), "../exportables/dashboard.html")):
        with open(filename, 'w') as dashboard:
            dashboard.write("<html><head></head><body>" + "\n")
            for fig in figs:
                inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                dashboard.write(inner_html)
            dashboard.write("</body></html>" + "\n")

    def add_history(self, with_cluster_0=False):
        if self.env.grep_type == 'greps':
            return

        df_stock = pd.read_csv(os.path.join(os.path.dirname(__file__), r'../exportables/simulate_stock_teco.csv'), parse_dates=['fecha'],
                               dtype={"geu": str, "stock": int, "centro": int}).rename({'centro': 'cram'}, axis=1)
        if self.env.grep_type == 'unicram':
            df_tickets = pd.read_csv(os.path.join(os.path.dirname(__file__), r'../exportables/simulate_tickets_teco_unicram.csv'),
                                     parse_dates=['fecha'], dtype={"geu": str, "Nivel_de_servicio": int, "cram": int})
        else:
            df_tickets = pd.read_csv(os.path.join(os.path.dirname(__file__), r'../exportables/simulate_tickets_teco.csv'),
                                     parse_dates=['fecha'], dtype={"geu": str, "Nivel_de_servicio": int, "cram": int})
            # df_travel = pd.read_csv(os.path.join(os.path.dirname(__file__), r'../exportables/travel_historico.csv'),
            #                         parse_dates=['fecha'],
            #                         dtype={"geu": str, "cram_suministrador": str, 'cram_receptor': str, 'cantidad': int})
        df_compras = pd.read_csv(os.path.join(os.path.dirname(__file__), r'../exportables/compras_historicas.csv'),
                                 parse_dates=['fecha'],
                                 dtype={"geu": str, "cantidad": int})
        df_entradas = pd.io.sql.read_sql("SELECT * from p02_entradas_by_date", con=postgres_connection(print_on=False), parse_dates=['fecha'])



        if self.geus_ids and not with_cluster_0:
            df_stock = df_stock[df_stock["geu"].isin(self.geus_ids)]
            df_tickets = df_tickets[df_tickets["geu"].isin(self.geus_ids)]
            df_compras = df_compras[df_compras["geu"].isin(self.geus_ids)]
            df_entradas = df_entradas[df_entradas["geu"].isin(self.geus_ids)]
            # if self.env.grep_type != 'unicram':
            #   df_travel = df_travel[df_travel["geu"].isin(self.geus_ids)]


        df_stock = df_stock[df_stock['fecha'] >= self.env.start_date]
        df_stock = df_stock[df_stock['fecha'] <= self.env.end_date]
        df_tickets = df_tickets[df_tickets['fecha'] >= self.env.start_date]
        df_tickets = df_tickets[df_tickets['fecha'] <= self.env.end_date]
        df_compras = df_compras[df_compras['fecha'] >= self.env.start_date]
        df_compras = df_compras[df_compras['fecha'] <= self.env.end_date]
        df_entradas = df_entradas[df_entradas['fecha'] >= self.env.start_date]
        df_entradas = df_entradas[df_entradas['fecha'] <= self.env.end_date]

        df_stock = df_stock[["geu", "cram", "fecha", "stock"]]
        df_stock['heuristica'] = 'historia'
        self.stock_data = self.stock_data.append(df_stock, ignore_index=True)

        df_tickets = df_tickets.groupby(['geu', 'cram', 'fecha'])["Nivel_de_servicio"].agg(
            ['sum', 'count']).reset_index()
        df_tickets.columns = ['geu', 'cram', "fecha", 'on_time', 'total']

        df_tickets["quincena"] = np.ceil(((df_tickets["fecha"] - self.env.start_date).dt.days)/ 15)
        df_tickets = df_tickets.groupby(["quincena", "geu", "cram"])[["on_time", "total"]].sum().reset_index()
        df_tickets["fecha"] = self.env.start_date + pd.to_timedelta(df_tickets["quincena"] * 15, unit="d")
        df_tickets.drop("quincena", axis=1, inplace=True)

        ids = [(geu, cram, fecha) for geu, cram, fecha in product(self.geus_ids, df_tickets['cram'].unique(),
                                                                  list(pd.date_range(start=self.env.start_date,
                                                                                     end=self.env.end_date, freq='15d'))
                                                                  )]
        df_tickets = df_tickets.set_index(["geu", "cram", "fecha"]).reindex(ids, fill_value=0).reset_index()

        df_tickets['heuristica'] = 'historia'
        self.service_data = self.service_data.append(df_tickets, ignore_index=True)

        df_compras['heuristica'] = 'historia'
        df_entradas['heuristica'] = 'historia'
        self.compras_data = self.compras_data.append(df_compras, ignore_index=True)
        self.entradas_data = self.entradas_data.append(self.process_historic_entradas(df_entradas), ignore_index=True)

        # if self.env.grep_type == '7_crams':
            # df_travel = df_travel[df_travel['fecha'] >= self.env.start_date]
            # df_travel = df_travel[df_travel['fecha'] <= self.env.end_date]
            # df_travel['mtime'] = df_travel.apply(lambda f: self.env.data_class.leadtime_between_warehouses[(f['cram_suministrador'],
            #                                                                                                f['cram_receptor'])], axis=1)
            # df_travel['weight'] = df_travel['geu'].apply(lambda f: self.env.data_class.geus_by_id[f].weight)
            # df_travel['mttime'] = df_travel['cantidad'] * df_travel['mtime']
            # df_travel['cost'] = df_travel.apply(lambda f: self.env.data_class.get_moving_price(f['weight']*f['cantidad'], f['mtime']), axis=1)

            # df_travel = df_travel.groupby(['fecha', 'geu']).agg({'weight': 'count', 'mtime': 'sum', 'mttime': 'sum', 'cost': 'sum'}).reset_index()
            # df_travel.rename({'weight': 'mlen'}, axis=1, inplace=True)
            # df_travel['heuristica'] = 'historia'
            # self.travel_data = self.travel_data.append(df_travel)

        if with_cluster_0:
            geus_in_cluster0 = [x for x, y in self.env.data_class.geus_by_id.items() if y.cluster == 0]
            df_stock = df_stock[df_stock["fecha"] == self.env.start_date]
            df_stock = df_stock[df_stock["geu"].isin(geus_in_cluster0)]
            fechas = [self.env.start_date + dt.timedelta(days=15 * n)
                      for n in range((floor((self.env.end_date - self.env.start_date).days / 15) + 2))]
            values = [(row["geu"], heu, fecha, row["cram"], row["stock"])
                   for _, row in df_stock.iterrows()
                   for heu in self.chosen_heuristics
                   for fecha in fechas]
            self.stock_data = self.stock_data.append(pd.DataFrame(values, columns=["geu", "heuristica", "fecha", "cram", "stock"]))

    def add_history_dash(self, history, with_cluster_0=False):

        df_stock = history['stock']
        df_tickets = history['tickets']

        if self.geus_ids and not with_cluster_0:
            df_stock = df_stock[df_stock["geu"].isin(self.geus_ids)]
            df_tickets = df_tickets[df_tickets["geu"].isin(self.geus_ids)]

        df_stock = df_stock[df_stock['fecha'] >= self.env.start_date]
        df_stock = df_stock[df_stock['fecha'] <= self.env.end_date]
        df_tickets = df_tickets[df_tickets['fecha'] >= self.env.start_date]
        df_tickets = df_tickets[df_tickets['fecha'] <= self.env.end_date]

        df_stock = df_stock[["geu", "centro", "fecha", "stock"]]
        df_stock['heuristica'] = 'historia'
        self.stock_data = self.stock_data.append(df_stock, ignore_index=True)

        df_tickets = df_tickets.groupby(['geu', 'centro', 'fecha'])["Nivel_de_servicio"].agg(
            ['sum', 'count']).reset_index()
        df_tickets.columns = ['geu', 'centro', "fecha", 'on_time', 'total']

        df_tickets["quincena"] = np.ceil(((df_tickets["fecha"] - self.env.start_date).dt.days) / 15)
        df_tickets = df_tickets.groupby(["quincena", "geu", "centro"])[["on_time", "total"]].sum().reset_index()
        df_tickets["fecha"] = self.env.start_date + pd.to_timedelta(df_tickets["quincena"] * 15, unit="d")
        df_tickets.drop("quincena", axis=1, inplace=True)

        ids = [(geu, cram, fecha) for geu, cram, fecha in product(self.geus_ids, df_tickets['centro'].unique(),
                                                                  list(pd.date_range(start=self.env.start_date,
                                                                                     end=self.env.end_date, freq='15d'))
                                                                  )]
        df_tickets = df_tickets.set_index(["geu", "centro", "fecha"]).reindex(ids, fill_value=0).reset_index()

        df_tickets['heuristica'] = 'historia'
        self.service_data = self.service_data.append(df_tickets, ignore_index=True)

        if with_cluster_0:
            geus_in_cluster0 = [x for x, y in self.env.data_class.geus_by_id.items() if y.cluster == 0]
            df_stock = df_stock[df_stock["fecha"] == self.env.start_date]
            df_stock = df_stock[df_stock["geu"].isin(geus_in_cluster0)]
            fechas = [self.env.start_date + dt.timedelta(days=15 * n)
                      for n in range((floor((self.env.end_date - self.env.start_date).days / 15) + 2))]
            values = [(row["geu"], heu, fecha, row["centro"], row["stock"])
                   for _, row in df_stock.iterrows()
                   for heu in self.chosen_heuristics
                   for fecha in fechas]
            self.stock_data = self.stock_data.append(pd.DataFrame(values, columns=["geu", "heuristica", "fecha", "centro", "stock"]))

    def save_dataframes(self):
        self.clean_stock()

        service_procesado = self.service_data.copy()
        service_procesado[['on_time_acumulado', "total_acumulado"]] = service_procesado.groupby(['heuristica', "cram", "geu"])[['on_time', "total"]].transform(
            pd.Series.cumsum)
        service_procesado.to_csv(os.path.join(os.path.dirname(__file__), "../exportables/service_simulado.csv"), index=False)

        self.stock_data.to_csv(os.path.join(os.path.dirname(__file__), "../exportables/stock_simulado.csv"), index=False)
        self.travel_data.to_csv(os.path.join(os.path.dirname(__file__), "../exportables/travel_simulado.csv"), index=False)
        self.entradas_data.to_csv(os.path.join(os.path.dirname(__file__), "../exportables/entradas_simulado.csv"), index=False)

        compras_procesado = self.compras_data.copy()
        compras_procesado['compras_acumulado'] = compras_procesado.groupby(['heuristica', 'geu'])["compras"].transform(pd.Series.cumsum)
        compras_procesado.to_csv(os.path.join(os.path.dirname(__file__), "../exportables/compras_simulado.csv"), index=False)

    def clean_stock(self):
        dates = [self.env.start_date + dt.timedelta(days=self.env.days_between_movements * i) for i in
                 range(floor((self.env.end_date - self.env.start_date).days / 15) + 1)]
        self.stock_data = self.stock_data[self.stock_data["fecha"].isin(dates)]

    def process_compras(self, df_compras):
        df_entradas = pd.io.sql.read_sql("SELECT * from p02_entradas", con=postgres_connection(print_on=False))
        df_entradas['pcompras'] = df_entradas["compras"] / (
                    df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])
        df_entradas['preparaciones'] = df_entradas["reparaciones"] / (
                df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])
        df_entradas['pdesmontes'] = df_entradas["desmontes"] / (
                df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])

        df_new_compras = pd.DataFrame(columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm", "ingresos_manuales"])
        for _, row in df_compras.iterrows():
            geu_id = row["geu"]
            if geu_id not in df_entradas["geu"].values:
                if self.env.data_class.geus_by_id[geu_id].procurement_type == "SPM":
                    df_new_compras = df_new_compras.append(
                        pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, 0, 0, row["compras"], 0)],
                                     columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                              "spm", "ingresos_manuales"]))
                elif self.env.data_class.geus_by_id[geu_id].procurement_type == "Buyable":
                    df_new_compras = df_new_compras.append(
                        pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"], 0, 0, 0, 0)],
                                     columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                              "spm", "ingresos_manuales"]))
                elif self.env.data_class.geus_by_id[geu_id].procurement_type == "Repairable":
                    df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, row["compras"], 0, 0, 0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                          "spm", "ingresos_manuales"]))
                else:
                    df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, 0, row["compras"], 0, 0)],
                    columns = ["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                               "spm", "ingresos_manuales"]))
            else:
                if self.env.data_class.geus_by_id[geu_id].procurement_type == "SPM":
                    df_new_compras = df_new_compras.append(pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"] *
                                                     df_entradas.loc[df_entradas['geu'] == row["geu"], 'pcompras'].iloc[0],
                                                     row["compras"] *
                                                     df_entradas.loc[df_entradas['geu'] == row["geu"], 'preparaciones'].iloc[0], 0,
                                                     row["compras"] *
                                                     df_entradas.loc[df_entradas['geu'] == row["geu"], 'pdesmontes'].iloc[0],
                                                     0)], columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm", "ingresos_manuales"]), ignore_index=True)
                elif self.env.data_class.geus_by_id[geu_id].is_dismountable:
                    df_new_compras = df_new_compras.append(
                        pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"] *
                                       df_entradas.loc[df_entradas['geu'] == row["geu"], 'pcompras'].iloc[0],
                                       row["compras"] *
                                       df_entradas.loc[df_entradas['geu'] == row["geu"], 'preparaciones'].iloc[0],
                                       row["compras"] *
                                       df_entradas.loc[df_entradas['geu'] == row["geu"], 'pdesmontes'].iloc[0], 0, 0)],
                                     columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                              "spm", "ingresos_manuales"]), ignore_index=True)
                else:
                    if df_entradas.loc[df_entradas['geu'] == row["geu"], 'preparaciones'].iloc[0] == 0:
                        df_new_compras = df_new_compras.append(
                            pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"],
                                           row["compras"],
                                           0, 0, 0, 0)],
                                         columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                                  "spm", "ingresos_manuales"]), ignore_index=True)
                    else:
                        df_new_compras = df_new_compras.append(
                            pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"],
                                           (row["compras"] *
                                            df_entradas.loc[df_entradas['geu'] == row["geu"], 'pcompras'].iloc[0]) / (
                                                       1 - df_entradas.loc[
                                                   df_entradas['geu'] == row["geu"], 'pdesmontes'].iloc[0]),
                                           (row["compras"] *
                                            df_entradas.loc[df_entradas['geu'] == row["geu"], 'preparaciones'].iloc[
                                                0]) / (1 - df_entradas.loc[
                                               df_entradas['geu'] == row["geu"], 'pdesmontes'].iloc[0]),
                                           0, 0, 0)],
                                         columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                                  "spm", "ingresos_manuales"]), ignore_index=True)
        return df_new_compras

    def process_historic_entradas(self, df_entradas):
        df_new_entradas = pd.DataFrame(columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm", "ingresos_manuales"])
        for _, row in df_entradas.iterrows():
            geu_id = row["geu"]
            if geu_id in self.env.data_class.geus_by_id and self.env.data_class.geus_by_id[geu_id].procurement_type == "SPM":
                df_new_entradas = df_new_entradas.append(pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"], row["reparaciones"], 0, row["desmontes"], 0)], columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm", "ingresos_manuales"]))
            elif geu_id in self.env.data_class.geus_by_id and self.env.data_class.geus_by_id[geu_id].is_dismountable == False:
                df_new_entradas = df_new_entradas.append(pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"],
                                                                        row["compras"], row["reparaciones"], 0, 0,
                                                                        row["desmontes"])],
                                                                      columns=["geu", "heuristica", "fecha", "compras",
                                                                               "reparaciones", "desmontes", "spm",
                                                                               "ingresos_manuales"]))
            else:
                df_new_entradas = df_new_entradas.append(pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"],
                                                                        row["compras"], row["reparaciones"],
                                                                        row["desmontes"], 0, 0)],
                                                                      columns=["geu", "heuristica", "fecha", "compras",
                                                                               "reparaciones", "desmontes", "spm",
                                                                               "ingresos_manuales"]), ignore_index=True)
        return df_new_entradas


if __name__ == "__main__":
    env_config = {"start_date": dt.datetime(2018, 10, 1), "end_date": dt.datetime(2020, 10, 15), "get_info": True,
                  'grep_type': 'greps', "from_pickle": False, 'use_historic_stocks': True, 'resupplies_per_year': 2,
                  "is_s4": False}
    model_vs_history = ModelVSHistory(env_config, chosen_heuristics=['Neural Net', 'Action1'])
    model_vs_history.run()
    if "Neural Net" in model_vs_history.chosen_heuristics:
        ray.shutdown()
    model_vs_history.add_history(with_cluster_0=True)
    model_vs_history.plot()
    model_vs_history.save_dataframes()
