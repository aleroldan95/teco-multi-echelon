from functions import elapsed_time_message, to_datetime
import time
import datetime as dt
from tqdm import tqdm
import pandas as pd
import os


class HistoryTeco:
    def __init__(self,
                 starting_date=dt.datetime(2018, 7, 1),
                 end_date=dt.datetime(2020, 10, 29),
                 ticket_sla=2,
                 data_class = None,
                 df_movements = None,
                 initial_stock = None):

        self.starting_date = starting_date
        self.end_date = end_date
        self.ticket_sla = ticket_sla
        self.data_class = data_class
        self.materials_by_id, self.geus_by_id, self.wh_list, self.conn, self.k_clusters = data_class.return_values()
        self.ticket_sla = ticket_sla

        # --------------------------------------------------------------------------------------------------------------

        self.process_movements(df_movements)

        # Hardcoded warehouse list and string_date
        self.warehouse_list = data_class.wh_list
        string_date = str(self.starting_date)[:10]

        # Initialize geus attibutes
        for geu in self.geus_by_id.values():
            geu.tickets_for_service_level = []
            geu.stock_by_day = {}
            geu.stock_by_wh = {}


        self.process_initial_stock_teco(initial_stock, string_date)

        # Clear tickets
        for material in self.materials_by_id.values():
            material.tickets = {}

        # Process Tickets
        #self.process_tickets_teco()

    # Methods ----------------------------------------------------------------------------------------------------------

    def run(self):
        # Simulate GEUs
        print('Simulating...')
        for geu in tqdm(self.geus_by_id.values()):
            self.simulate_geu(geu)

        # Show results and generate exportable dataframe
        print('\n Final Result:\n')
        self.show_final_result()


    def process_movements(self, movements):
        # Start Timer
        tic = time.time()
        message = "Set Historic Movements..."
        # ------------------------------------------------------------------------------------------------------------------
        greps_banned = [wh for wh in list(movements['ce_alm_s44'].unique()) if
                        wh not in self.data_class.relation_grep_wh.keys()]
        # if greps_banned:
        #    st.warning(
        #        f"Las siguientes relaciones de centro-almacén no se encontraron en la base de datos relacion_grep_almacen: {greps_banned}. "
        #        f"No serán tenidos en cuenta. Para que sean incluidos se deben agregar dichas relaciones con su grep correspondiente en relacion_grep_alamcen")
        df_movements = movements[~movements['ce_alm_s44'].isin(greps_banned)]

        df_movements['ce_alm_s44'] = df_movements['ce_alm_s44'].map(
            lambda x: self.data_class.relation_grep_wh[x])

        # Load and process movements
        df_movements = df_movements.groupby(["material", "fecha", "ce_alm_s44"])["cantidad"].sum().reset_index()
        # Convert date column to datetime
        df_movements["fecha"] = pd.to_datetime(df_movements["fecha"])

        df_movements = df_movements[df_movements["fecha"] > self.starting_date]

        [self.materials_by_id[material].set_movements_for_teco_simulation(to_datetime(date), cram, amount)
         for material, date, cram, amount in df_movements.values
         if material in self.materials_by_id.keys()]

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

    def process_initial_stock_teco(self, initial_stock, string_date):
        # Start Timer
        tic = time.time()
        message = "Set starting stock..."
        # ------------------------------------------------------------------------------------------------------------------
        # Process stock & tickets
        greps_banned = [wh for wh in list(initial_stock['ce_alm_s44'].unique()) if
                        wh not in self.data_class.relation_grep_wh.keys()]
        # if greps_banned:
        #    st.warning(
        #        f"Las siguientes relaciones de centro-almacén no se encontraron en la base de datos relacion_grep_almacen: {greps_banned}. "
        #        f"No serán tenidos en cuenta. Para que sean incluidos se deben agregar dichas relaciones con su grep correspondiente en relacion_grep_alamcen")
        df_initial_stock = initial_stock[~initial_stock['ce_alm_s44'].isin(greps_banned)]

        df_initial_stock['ce_alm_s44'] = df_initial_stock['ce_alm_s44'].map(
            lambda x: self.data_class.relation_grep_wh[x])

        df_initial_stock.columns = ['material', 'fecha', 'stock', 'grep']

        # Set default initial stock as 0
        for geu in self.geus_by_id.values():
            geu.set_zero_stock(self.warehouse_list)

        # Set starting stock for the given initial date
        for index, row in df_initial_stock.iterrows():
            if row['material'] in self.materials_by_id.keys():
                self.materials_by_id[row['material']].geu.set_starting_stock_from_row(row, is_teco_simulation=True)

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)


    def simulate_geu(self, geu):
        today = self.starting_date
        outflow_movements = []

        while today != self.end_date:
            # Daily Internal Movements
            sim_daily_movements(geu, today)

            # Outbound Movements - Update tickets
            outflow_movements = geu.sim_update_tickets_for_teco_simulation(today, outflow_movements, self.ticket_sla)

            geu.stock_by_day[today] = dict(geu.stock_by_wh)

            today = today + dt.timedelta(days=1)

    def check_non_negative_stock(self):
        for geu in self.geus_by_id.values():
            for date, stock_by_wh in geu.stock_by_day.items():
                is_non_negative = all([stock >= 0 for stock in stock_by_wh.values()])
                if not is_non_negative:
                    print(f"\t⚠ [Warning]\tGEU: {geu.id} with negative stock at {date}")

        print("\n✓ [Checked]")


    def show_final_result(self):

        # Initialize dict
        service_level_per_wh = {}

        # Iterate through all warehouses
        for wh in self.warehouse_list:
            dates = []
            list_y = []
            acum = []

            # Iterate through all geus
            for geu in self.geus_by_id.values():
                for t in geu.tickets_for_service_level:
                    if t.grep == wh:
                        if t.is_closed and not t.is_broken:
                            y = 1
                        else:
                            y = 0

                        dates.append(t.ticket_date)
                        list_y.append(y)
                        acum.append(sum(list_y) / len(list_y))

            if acum:
                service_level_per_wh[wh] = acum[-1:][0]

        # Print results
        for k, v in service_level_per_wh.items():
            print(f'\t<CRAM {k}>: {round(v*100, 2)}%')

    def generate_exportables_dataframes(self):
        # Start Timer
        message = "Generating exportable file..."
        tic = time.time()
        # ------------------------------------------------------------------------------------------------------------------

        # Initiate information lists
        today = self.starting_date
        stock = []
        dates = []
        geus = []
        cram = []
        cluster = []
        #exportables_geus = [geu for geu in self.geus_by_id.values() if geu.tickets]

        # Fill info lists
        last = False
        while today != self.end_date and not last:
            if today >= self.end_date:
                today = self.end_date - dt.timedelta(days=1)
                last = True
            for geu in self.geus_by_id.values():
                for wh in geu.stock_by_wh.keys():
                    dates.append(today)
                    stock.append(geu.stock_by_day[today][wh])
                    geus.append(geu.id)
                    cram.append(wh)
                    cluster.append(geu.cluster)

            today += dt.timedelta(days=15)

        # Convert lists to DataFrame
        df_stocks = pd.DataFrame({'geu': geus, 'fecha': dates, 'centro': cram, 'stock': stock, 'cluster':cluster})

        # ------------------Tickets------------------

        # Initiate information lists
        is_closed = []
        is_broken = []
        service_level = []
        dates = []
        geus = []
        cram = []
        cluster = []

        # Fill info lists
        for geu in self.geus_by_id.values():
            for t in geu.tickets_for_service_level:
                dates.append(t.ticket_date)
                is_closed.append(int(t.is_closed))
                is_broken.append(t.is_broken)
                service_level.append(int(t.is_closed and not t.is_broken))
                geus.append(geu.id)
                cram.append(t.grep)
                cluster.append(geu.cluster)

        # Convert lists to DataFrame
        df_tickets = pd.DataFrame({'geu': geus, 'fecha': dates, 'centro': cram, 'Quebrado': is_broken,
                                   'Cerrado': is_closed, 'Nivel_de_servicio': service_level,
                                   'cluster': cluster})


        df_tickets['fecha'] = pd.to_datetime(df_tickets['fecha'])
        df_stocks['fecha'] = pd.to_datetime(df_stocks['fecha'])
        # ------------------Exports------------------

        # Export information to csv
        #my_path = os.path.join(os.path.dirname(__file__), 'dashboard_csv/')
        # df_stocks.to_csv(f'{my_path}simulate_stock_teco.csv')
        #df_tickets.to_csv(f'{my_path}simulate_tickets_teco.csv')

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

        return {'tickets': df_tickets, 'stock': df_stocks}

# ----------------------------------------------------------------------------------------------------------------------

def sim_daily_movements(geu, today):
    for wh in geu.stock_by_wh.keys():
        try:
            geu.stock_by_wh[wh] += geu.movements_for_teco_simulation[(today, wh)]
        except:
            pass

# ----------------------------------------------------------------------------------------------------------------------


