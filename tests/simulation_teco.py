from functions import main_timer, create_query, postgres_connection, elapsed_time_message, to_datetime
import time
import pyfiglet
import datetime as dt
from classes.ticket import Ticket
from tqdm import tqdm
from data_loader_class import DataClass
import pandas as pd
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import os


class HistoryTeco:

    def __init__(self,
                 starting_date=dt.datetime(2018, 7, 1),
                 end_date=dt.datetime(2020, 10, 29),
                 ticket_sla=2,
                 grep_type='7_crams'):

        self.starting_date = starting_date
        self.end_date = end_date
        self.ticket_sla = ticket_sla
        self.grep_type = grep_type
        data_class = DataClass(print_on=True, is_streamlit=False, grep_type=grep_type, start_date=starting_date, is_s4=False)
        self.materials_by_id, self.geus_by_id, self.wh_list, self.conn, self.k_clusters = data_class.return_values()
        self.ticket_sla = ticket_sla

        # --------------------------------------------------------------------------------------------------------------
        # Start main timer
        tic = main_timer(True)

        # Figlet print
        custom_fig = pyfiglet.Figlet(font='slant')
        print("\n\nWelcome to")
        print(custom_fig.renderText('Teco Simu'))

        """ # Initialize materials demands
        for material in self.materials_by_id.values():
            material.demands = []

        # Load and process demand
        df_demand = self.load_demand()
        self.process_demand(df_demand)"""

        # Load and process movements
        df_movements = self.load_movements()
        self.process_movements(df_movements)

        # Hardcoded warehouse list and string_date
        self.warehouse_list = data_class.wh_list
        string_date = str(self.starting_date)[:10]

        # Initialize geus attibutes
        for geu in self.geus_by_id.values():
            geu.tickets_for_service_level = []
            geu.stock_by_day = {}
            geu.stock_by_wh = {}

        # Process stock & tickets
        self.process_initial_stock_teco(string_date)

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

    def validate_stock(self):
        print('\nTesting positive stock:\n')
        self.check_non_negative_stock()

    def load_demand(self):
        """
        Generate a dataframe with materials' demand
        :param engine: DB connection
        :return: pd.DataFrame with demand history
        """
        # Start Timer
        tic = time.time()
        message = "Loading Demand Data..."
        # ------------------------------------------------------------------------------------------------------------------

        # Query: demand
        query = """select * from p02_daily_demand_almacen where centro not like 'SE%%';"""
        df_demand = create_query(query, self.conn)
        # Convert date column to datetime
        df_demand["fecha"] = pd.to_datetime(df_demand["fecha"])
        # Prepare data
        df_demand["cram"] = df_demand["centro"].map(lambda x: x[2:])
        # Reduce demand to bare minimum
        df_demand = df_demand.groupby(["fecha", "material", "cram"])["cantidad"].sum().reset_index()

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)
        return df_demand

    def process_demand(self, df_demand):
        """
        Adding Material's demand
        :param df_demand:
        :param materials_by_id:
        :return:
        """
        # Start Timer
        message = "Processing Demand..."
        tic = time.time()
        # ------------------------------------------------------------------------------------------------------------------

        [self.materials_by_id[material].add_demand(to_datetime(date), warehouse, demand)
         for date, material, warehouse, demand in df_demand.values
         if material in self.materials_by_id]

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

    def process_movements(self, df_movements):
        # Start Timer
        tic = time.time()
        message = "Set Historic Movements..."
        # ------------------------------------------------------------------------------------------------------------------

        [self.materials_by_id[material].set_movements_for_teco_simulation(to_datetime(date), cram, amount)
         for material, date, cram, amount in df_movements.values
         if material in self.materials_by_id.keys()]

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

    def process_initial_stock_teco(self, string_date):
        # Start Timer
        tic = time.time()
        message = "Set starting stock..."
        # ------------------------------------------------------------------------------------------------------------------

        # Query: initial stock (matching the string_date)
        query = f"""SELECT material, fecha, centro, stock 
                        FROM p02_stock_by_date 
                        where fecha = '{string_date}' and centro like 'C%%';"""
        starting_stock_df = create_query(query, self.conn)
        if self.grep_type == 'unicram': starting_stock_df['centro'] = 'C00'

        # Set default initial stock as 0
        for geu in self.geus_by_id.values():
            geu.set_zero_stock(self.warehouse_list)

        # Set starting stock for the given initial date
        for index, row in starting_stock_df.iterrows():
            if row['material'] in self.materials_by_id.keys():
                self.materials_by_id[row['material']].geu.set_starting_stock_from_row(row, is_teco_simulation=True)

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

    def process_tickets_teco(self):
        """
        Set GEU tickets for simulation tests
        :param engine: DB connection
        :param materials_by_id: {id, Material}
        :param starting_date: dt.date
        :return: void -> Complete GEU object info
        """
        # Start Timer
        tic = time.time()
        message = "Set tickets for sim test..."
        # ------------------------------------------------------------------------------------------------------------------

        # Load tickets
        query = f"""select * from p02_tickets_by_cram;"""
        df_tickets = create_query(query, self.conn)
        if self.grep_type == 'unicram': df_tickets["cram"] = 'C00'

        # Convert date columns to datetime
        df_tickets["fecha"] = pd.to_datetime(df_tickets["fecha"])

        # Add tickets demands for greps without movements demands
        [self.materials_by_id[str(material)].add_demand(to_datetime(date), cram, int(amount))
         for date, material, amount, cram in df_tickets[["fecha", "material", "cantidad", "cram"]].values
         if str(material) in self.materials_by_id.keys() and
         cram not in set([cram for date, cram, amount in self.materials_by_id[str(material)].demands])]

        # Create Ticket class
        df_tickets = df_tickets[["fecha", "material", "cantidad", "cram"]]
        df_tickets.columns = ["fecha", "material", "cantidad", "grep"]
        [Ticket(row, self.materials_by_id[str(row["material"])])
         for iter, row in df_tickets.iterrows()
         if str(row["material"]) in self.materials_by_id.keys()]

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

    def load_movements(self):
        # Start Timer
        tic = time.time()
        message = "Load Historic Movements..."
        # ------------------------------------------------------------------------------------------------------------------

        # Query: load movements
        query = """SELECT * FROM p01_historic_movements_for_teco_simulation;"""
        df_movements = create_query(query, self.conn)
        # Preparing data
        df_movements["cram"] = df_movements["centro"].map(lambda x: x[2:])
        if self.grep_type=='unicram': df_movements["cram"] = '00'
        # Reduce movements to bare minimum
        df_movements = df_movements.groupby(["material", "fecha_de_documento", "cram"])["cantidad"].sum().reset_index()
        # Convert date column to datetime
        df_movements["fecha_de_documento"] = pd.to_datetime(df_movements["fecha_de_documento"])

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

        return df_movements

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
        """
        # Initiate information lists
        today = self.starting_date
        stock = []
        dates = []
        geus = []
        cram = []
        cluster = []
        #exportables_geus = [geu for geu in self.geus_by_id.values() if geu.tickets]

        # Fill info lists
        while today != self.end_date:
            for geu in self.geus_by_id.values():
                for wh in geu.stock_by_wh.keys():
                    dates.append(today)
                    stock.append(geu.stock_by_day[today][wh])
                    geus.append(geu.id)
                    cram.append(wh)
                    cluster.append(geu.cluster)

            today += dt.timedelta(days=1)

        # Convert lists to DataFrame
        df_stocks = pd.DataFrame({'geu': geus, 'fecha': dates, 'cram': cram, 'stock': stock, 'cluster':cluster})
        """
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
        df_tickets = pd.DataFrame({'geu': geus, 'fecha': dates, 'cram': cram, 'Quebrado': is_broken,
                                   'Cerrado': is_closed, 'Nivel_de_servicio': service_level,
                                   'cluster': cluster})

        # ------------------Exports------------------

        # Export information to csv
        my_path = os.path.join(os.path.dirname(__file__), '../exportables/')
        # df_stocks.to_csv(f'{my_path}simulate_stock_teco.csv')
        df_tickets.to_csv(f'{my_path}simulate_tickets_teco.csv')

        # ------------------------------------------------------------------------------------------------------------------
        # Stop Timer
        toc = time.time()
        elapsed_time_message(toc - tic, message)

    def plot_result(self):

        print('Plotting...')
        # Stocks
        my_path = os.path.join(os.path.dirname(__file__), '../exportables/')
        df_stock = pd.read_csv(f'{my_path}simulate_stock_teco.csv')
        df_stock.fecha = pd.to_datetime(df_stock.fecha)
        df_stock_cluster = df_stock.groupby(['fecha', 'cluster']).sum().reset_index()
        df_stock_cram = df_stock.groupby(["fecha", "cram"])["stock"].sum().reset_index()

        # Tickets
        df_tickets_00 = pd.read_csv(f'{my_path}/simulate_tickets_teco.csv')
        df_tickets_00.fecha = pd.to_datetime(df_tickets_00.fecha)
        df_tickets = df_tickets_00.groupby(['fecha', 'cram'])["Nivel_de_servicio"].agg(['sum', 'count'])
        df_tickets.reset_index(inplace=True)
        df_tickets.columns = ["fecha", 'cram', 'suma', 'cantidad']
        df_tickets_cluster = df_tickets_00.groupby(['fecha', 'cluster'])["Nivel_de_servicio"].agg(['sum', 'count'])
        df_tickets_cluster.reset_index(inplace=True)
        df_tickets_cluster.columns = ["fecha", 'cluster', 'suma', 'cantidad']

        # Initialize plot
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=(f'Evolución de Stock',
                                            f'Evolución de Nivel de Servicio Acumulado'),
                            vertical_spacing=0.12)

        # Plots---------------------------------------------------------------------------------------------------------
        # Stock plot
        subset = df_stock.groupby(["fecha"])["stock"].sum().reset_index()
        subset.columns = ["fecha", "stock"]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='Stock Total',
                                 mode='lines',
                                 line_color='black',
                                 visible=False),
                      row=1, col=1)

        # List of colors
        list_of_colors = px.colors.qualitative.Dark24
        dictionary_of_colors = {}
        for i, cram in enumerate(list(df_stock.cram.unique())):
            dictionary_of_colors[cram] = list_of_colors[i]

        for cram in list(df_stock.cram.unique()):
            subset = df_stock_cram[df_stock_cram.cram == cram]
            fig.add_trace(go.Scatter(x=subset.fecha,
                                     y=subset.stock,
                                     name='<Stock - CRAM> ' + str(cram),
                                     mode='lines',
                                     line_color=dictionary_of_colors[cram],
                                     visible=True),
                          row=1, col=1)

        # Tickets Plot
        # Total
        subset = df_tickets_00.groupby(['fecha'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
        subset.columns = ["fecha", 'suma', 'cantidad']
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                 mode='lines',
                                 name='Nivel de Servicio',
                                 line_color='black',
                                 visible=False),
                      row=2, col=1)
        # list(df_tickets.cram.unique())

        for cram in list(df_stock.cram.unique()):
            subset = df_tickets[df_tickets.cram == cram]
            fig.add_trace(go.Scatter(x=subset.fecha,
                                     y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                     mode='lines',
                                     name=f'<SL - CRAM> {cram}',
                                     visible=True,
                                     line_color=dictionary_of_colors[cram]),
                          row=2, col=1)

        # Cluster Stock
        for cluster in list(df_stock.cluster.unique()):
            subset = df_stock_cluster[df_stock_cluster.cluster == cluster]
            fig.add_trace(go.Scatter(x=subset.fecha,
                                     y=subset.stock,
                                     name='<Stock - CLUSTER> ' + str(cluster),
                                     mode='lines',
                                     line_color=dictionary_of_colors[cluster],
                                     visible=False),
                          row=1, col=1)

        # Cluster Tickets
        for cluster in list(df_tickets_cluster.cluster.unique()):
            subset = df_tickets_cluster[df_tickets_cluster.cluster == cluster]
            fig.add_trace(go.Scatter(x=subset.fecha,
                                     y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                                     mode='lines',
                                     name=f'<SL - CLUSTER> {cluster}',
                                     visible=False,
                                     line_color=dictionary_of_colors[cluster]),
                          row=2, col=1)

        # Buttons-------------------------------------------------------------------------------------------------------

        button1 = dict(method='update',
                       args=[{"visible": [False,                                                # Total stock
                                          True, True, True, True, True, True, True,            # Stock CRAMs
                                          False,                                                # Total Service Level
                                          True, True, True, True, True, True, True,             # Service Level CRAMs
                                          False, False, False, False, False,                    # Stock cluster
                                          False, False, False, False, False]}],                 # Service Level cluster
                       label="<CRAM>")

        button2 = dict(method='update',
                       args=[{"visible": [False,
                                          False, False, False, False, False, False, False,
                                          False,
                                          False, False, False, False, False, False, False,
                                          True, True, True, True, True,
                                          True, True, True, True, True]}],
                       label="<CLUSTER>")

        button3 = dict(method='update',
                       args=[{"visible": [True,
                                          False, False, False, False, False, False, False,
                                          True,
                                          False, False, False, False, False, False, False,
                                          False, False, False, False, False,
                                          False, False, False, False, False]}],
                       label="<TOTAL>")

        # LayOut-------------------------------------------------------------------------------------------------------
        # Update Layout
        fig.update_layout(
            updatemenus=[dict(type='buttons',
                              buttons=[button1, button2, button3],
                              x=1.05,
                              xanchor="left",
                              y=0.2,
                              yanchor="top")])

        fig.update_yaxes(title_text="Stock", row=1, col=1)

        fig.update_xaxes(title_text="Month",
                         row=2, col=1)
        fig.update_yaxes(title_text="Service Level", row=2, col=1)

        plotly.offline.plot(fig, filename=f'model_vs_history_new_scope.html')

# ----------------------------------------------------------------------------------------------------------------------


def sim_daily_movements(geu, today):
    for wh in geu.stock_by_wh.keys():
        try:
            geu.stock_by_wh[wh] += geu.movements_for_teco_simulation[(today, wh)]
        except:
            pass

# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    HistoryTeco = HistoryTeco()
    HistoryTeco.run()
    HistoryTeco.generate_exportables_dataframes()
    #HistoryTeco.plot_result()
