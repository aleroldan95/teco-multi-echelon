import numpy as np
import pandas as pd
import math
import datetime as dt
import os
import streamlit as st

from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import networkx as nx

from classes.ticket import Ticket
from classes.period import Period
from functions import linear_forecast, expo_forecast


class GEU:
    # Constructors------------------------------------------------------------------------------------------------------

    def __init__(self, group_id, list_of_materials):

        # Starting attributes
        self.id = group_id
        self.materials = set()
        for mat in list_of_materials:
            self.materials.add(mat)

        # Attributes to be defined by heuristics taking the materials inside each GEU
        # Said heuristics are applied in the set_parameters() function

        # Unique value among materials
        self.domain = None
        self.domain_override = False
        self.has_multiple_domains = False
        self.name = None
        self.name_override = None
        self.equipment = None
        self.equipments = []
        self.brand = None
        self.area = None
        self.subarea = None

        # Price
        self.price = None
        self.price_type = None
        self.price_unit = None
        self.rep_price = None
        self.rep_unit = None

        # Criticality
        self.criticality = None
        self.has_criticality_data = True
        self.set_criticality()

        # Leadtimes
        self.leadtime = None
        self.leadtime_sd = None
        self.leadtime_override = False
        self.weight = None
        self.weight_override = False

        # Complete objects information
        self.complete_material_equivalences()
        self.demands = []
        self.demand_priority = []

        # Old attributes
        self.catalog = group_id
        self.add_geu_relationship(list_of_materials)
        self.is_unigeu = None

        # Graph
        self.graph = None
        self.is_strongly_connected = None

        # Procurement
        self.is_buyable = None
        self.is_repairable = None
        self.is_dismountable = None
        self.is_spm = None
        self.has_procurement_type = None
        self.procurement_type = None

        # Clustering
        self.cluster = None
        self.weighted_demand_v1_tkt = None

        # Simulation attributes
        self.stock_by_wh: {str: int} = {}  # End-of-the-day value
        self.stock_in_transit: {str: int} = {}  # End-of-the-day value
        self.tickets: {dt.date, Ticket} = {}  # Full history of tickets by date

        # Simulation Teco
        self.tickets_for_service_level = []
        self.movements_for_teco_simulation = {}
        self.stock_by_day = {}
        self.starting_stock_by_wh = {}
        self.stock_in_transit_by_wh = {}

    # ==================================================================================================================

    def __repr__(self):
        name = "<Geu> group_id:{}".format(self.id) + '\t' + "materials:[".format(self.id)
        q_mat = len(self.materials)
        i = 1
        for material in self.materials:
            name += material.catalog
            if i < q_mat:
                name += ", "
                i += 1
            else:
                name += "]"
        return name

    # Setters-----------------------------------------------------------------------------------------------------------

    def complete_material_equivalences(self):
        """
        Set material equivalences and geu for each material composing the GEU
        :return: Void -> complete material object's attributes
        """
        for material in self.materials:
            material.geu = self
            for material_aux in self.materials:
                material.equivalent_materials.add(material_aux)

    def add_material_equivalence(self, mat):
        self.materials.add(mat)
        self.complete_material_equivalences()

    def complete_object_information(self, accepted_domains: list, df_lt_spm: pd.DataFrame,
                                    df_lt_repair: pd.DataFrame, df_name_override: pd.DataFrame):
        """
        Complete GEU information for attributes varying between different materials
        :return: VOID -> complete GEU info
        """

        # -----------------------------

        # Dependent variables

        # Set Domain
        for mat in self.materials:
            # Set domain
            if self.domain is None and mat.domain in accepted_domains:
                self.domain = mat.domain
            # Check if domain is not unique
            elif self.domain != mat.domain:
                self.has_multiple_domains = True
        # If it's still empty, there is no accepted domain in the GEU
        if self.domain is None:
            raise ValueError('ERROR: El GEU no tiene ningún dominio válido.')

        # Set Brand
        # Analog to Domain, see above
        for mat in self.materials:
            if self.brand is None:
                self.brand = mat.brand
            elif self.brand != mat.brand:
                self.brand = 'Many'

        # Order Conditions and lead times
        self.set_order_conditions(df_lt_spm, df_lt_repair)

        # Price
        # Select best method
        self.price_type = '0-Mean'
        for mat in self.materials:
            if int(mat.price_type[0]) > int(self.price_type[0]):
                self.price_type = mat.price_type

        # Get all prices of best method
        price_to_use = 0
        if self.is_buyable:
            for mat in self.materials:
                if mat.price_type == self.price_type:
                    price_to_use = max(price_to_use, mat.price)
        else:
            prices = 0
            for mat in self.materials:
                if mat.price_type == self.price_type:
                    price_to_use += mat.price
                    prices += 1
            price_to_use /= prices
        self.price = int(price_to_use)

        self.rep_price = int(max([mat.rep_price for mat in self.materials]))

        self.price_unit = "USD"
        self.rep_unit = "ARS"

        # -----------------------------

        # Independent variables

        # Set Name
        # Take description of first material in the GEU
        if self.id in df_name_override.index:
            new_name = df_name_override.loc[self.id]['Nombre']
            if new_name is not None and len(new_name) > 0:
                self.name = new_name
            else:
                self.name = list(self.materials)[0].name
        else:
            self.name = list(self.materials)[0].name

        # Set Equipment
        # Analog to Domain, see above
        for mat in self.materials:
            if mat.equipment:
                if self.equipment is None:
                    self.equipment = mat.equipment
                else:
                    self.equipment += '/' + mat.equipment

                for equip in mat.equipments:
                    if equip not in mat.equipments:
                        self.equipments.append(equip)

        # Set Area
        # Analog to Domain, see above
        for mat in self.materials:
            if self.area is None:
                self.area = mat.area
            elif self.area != mat.area:
                self.area = 'Many'

        # Set Subarea
        # Analog to Domain, see above
        for mat in self.materials:
            if self.subarea is None:
                self.subarea = mat.subarea
            elif self.subarea != mat.subarea:
                self.subarea = 'Many'

        # Set Criticality
        # Take highest criticality amongst GEU's materials
        list_aux = [material.criticality for material in self.materials]
        if 'critico' in list_aux:
            self.criticality = 'critico'
        elif 'mayor' in list_aux:
            self.criticality = 'mayor'
        elif 'bajo' in list_aux:
            self.criticality = 'bajo'
        else:
            self.criticality = 'bajo'

        # Set weight
        mats_weights = [mat.weight for mat in self.materials if mat.weight is not None]
        if mats_weights:
            self.weight = np.mean(mats_weights)
        else:
            self.weight = 1
            self.weight_override = True

    def set_ticket(self, ticket):
        if ticket.ticket_date not in self.tickets.keys():
            self.tickets[ticket.ticket_date] = []
            self.tickets[ticket.ticket_date].append(ticket)
        else:
            self.tickets[ticket.ticket_date].append(ticket)

    def set_demands(self):
        for material in self.materials:
            self.demands.extend(material.demands)

    def update_demand_priorities(self, wh_list, date):
        demand_by_wh = {wh: 0 for wh in wh_list}
        for demand in self.demands:
            if demand[0] < date and demand[0] >= date - dt.timedelta(days=365):
                demand_by_wh[demand[1]] += demand[2]
        self.demand_priority = sorted(demand_by_wh, key=lambda wh: demand_by_wh[wh])

    def get_first_date_demand(self, warehouse):
        if self.demands:
            if warehouse == 'all':
                return min([dates for dates, warehouse, demand in self.demands])
            list_of_demands = [dates for dates, ware_house, demand in self.demands if ware_house == warehouse]
            return min(list_of_demands) if len(list_of_demands) != 0 else None

    def set_movements_for_teco_simulation(self, date, cram, amount):
        if (date, cram) in self.movements_for_teco_simulation.keys():
            self.movements_for_teco_simulation[(date, cram)] += amount
        else:
            self.movements_for_teco_simulation[(date, cram)] = amount

    # ==================================================================================================================

    # Getters-----------------------------------------------------------------------------------------------------------

    def get_demand(self, period_object, warehouse):
        if warehouse == 'all':
            list_of_amounts = [amount for date, ware_house, amount in self.demands if period_object.contains(date)]
        else:
            list_of_amounts = [amount for date, ware_house, amount in self.demands if
                               period_object.contains(date) and
                               ware_house == warehouse]

        return sum(list_of_amounts)

    def get_forecast(self, period_object, warehouse, time_delta=relativedelta(years=3), soft=False, type='expo'):

        first_movement = self.get_first_date_demand(warehouse)

        if first_movement is None:
            return 0

        if first_movement < period_object.first_date:
            if type == 'linear':
                result = linear_forecast(self, period_object,
                                         first_date=max(period_object.first_date - time_delta, first_movement),
                                         warehouse=warehouse, soft=soft)
            elif type == 'expo':
                result = expo_forecast(self, period_object,
                                       first_date=max(period_object.first_date - time_delta, first_movement),
                                       warehouse=warehouse, soft=soft)
            else:
                raise NotImplemented('Type of forecast not implemented: ' + type)
            return max(float(result), 0)
        else:
            return 0

    def set_starting_stock_from_row(self, starting_stock_df_row, is_teco_simulation=False):
        """
            Set starting stock for simulation run
            :param starting_stock_df_row: row of starting stock DataFrame
            :param is_teco_simulation:
            :return: void -> complete geu attribute
        """
        if not is_teco_simulation:
            wh = starting_stock_df_row['grep']
        else:
            wh = starting_stock_df_row['grep']

        stock = starting_stock_df_row['stock']
        # self.starting_stock_by_wh[wh] = self.starting_stock_by_wh[wh] + stock
        self.stock_by_wh[wh] = self.stock_by_wh[wh] + stock

    def set_zero_stock(self, warehouse_list):
        """
        Set default value of stock as 0
        :param warehouse_list: list of active warehouses
        :return: void -> complete geu attribbte
        """
        for wh in warehouse_list:
            self.starting_stock_by_wh[wh] = 0
            self.stock_by_wh[wh] = 0
            self.stock_in_transit_by_wh[wh] = 0

    def average_service_level_per_grep_simulation(self, finish_date=dt.datetime(2020, 6, 14), is_teco_simulation=False):

        if not is_teco_simulation:
            tickets_for_service_level = self.simulation_runs[0].tickets_for_service_level
        else:
            tickets_for_service_level = self.tickets_for_service_level

        service_level_per_wh = {}
        for wh in self.stock_by_wh.keys():
            dates = []
            list_y = []
            acum = []
            # last_day = max([ticket.ticket_date for ticket in self.tickets_for_service_level])

            for t in tickets_for_service_level:
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

        if is_teco_simulation:
            return service_level_per_wh

        for k, v in service_level_per_wh.items():
            print(f'\t<CRAM {k}>: {round(v, 2)}%')

    def sim_update_tickets_for_teco_simulation(self, today, outflow_movements, ticket_sla):

        # Fetch today's tickets
        if today in self.tickets.keys():
            for tickets_today in self.tickets[today]:
                outflow_movements.append(tickets_today)
                self.tickets_for_service_level.append(tickets_today)

        # Failed tickets
        for ticket in outflow_movements:
            delta = today - ticket.ticket_date
            if delta.days >= ticket_sla:
                if not ticket.is_closed:
                    ticket.is_broken = True
                    # outflow_movements.remove(ticket)

        # Fulfill tickets
        for ticket in outflow_movements:
            if not ticket.is_closed:
                if self.stock_by_wh[ticket.grep] >= ticket.amount:
                    # Fulfills ticket
                    self.stock_by_wh[ticket.grep] -= ticket.amount_left
                    ticket.is_closed = True
                    # outflow_movements.remove(ticket)
                else:
                    # Partially Fulfills ticket
                    ticket.amount_left = ticket.amount_left - self.stock_by_wh[ticket.grep]
                    self.stock_by_wh[ticket.grep] = 0
                    ticket.is_partial_closed = True

        return outflow_movements

    # Old GEU methods

    def add_geu_relationship(self, list_of_materials):
        for material in list_of_materials:
            material.set_geu(self, list_of_materials)

    def size(self):
        return len(self.materials)

    def get_stock(self, date=dt.datetime(2020, 1, 1)):
        return sum([material.get_stock(date) for material in self.materials])

    def get_stock_with_transit(self, date):
        return sum([material.get_stock_with_transit(date) for material in self.materials])

    def set_order_conditions(self, df_lt_spm: pd.DataFrame, df_lt_repair: pd.DataFrame, procurement_mode: int = 1):
        """
        Sets an unique procurement type, leadtime mean and std for the GEU
        :return:
        """

        # Set procurement types based on any match found
        self.is_spm = any([material.is_spm for material in self.materials])
        self.is_repairable = any([material.is_repairable for material in self.materials])
        self.is_buyable = any([material.is_buyable for material in self.materials])
        self.is_dismountable = any([material.is_dismountable for material in self.materials])

        # If no procurement type set as buyable
        self.has_procurement_type = self.is_dismountable or self.is_buyable or self.is_spm or self.is_repairable
        if not self.has_procurement_type:
            self.is_buyable = True

        # --------------

        # Set unique values (and override if needed)

        # If CORE VOZ, set dismountable instead of repairable
        if procurement_mode == 0:
            self.procurement_type = 'Buyable'
            self.leadtime = max([material.leadtime for material in self.materials if material.is_buyable])
            self.leadtime_sd = max([material.leadtime_sd for material in self.materials if material.is_buyable])
            return
        elif procurement_mode == 1:
            if self.domain == 'CORE VOZ' and not self.is_spm and not self.is_buyable\
                    and self.is_dismountable and self.is_repairable:
                self.procurement_type = 'Dismountable'
                self.leadtime = 90
                self.leadtime_sd = 0
                return

            if self.is_spm:
                self.procurement_type = 'SPM'

                # Override
                if (self.domain, self.brand) in df_lt_spm.index:
                    try:
                        new_leadtime = df_lt_spm.loc[(self.domain, self.brand)]['leadtime_spm']
                        new_leadtime = float(new_leadtime)

                        self.leadtime_override = True
                        self.leadtime = new_leadtime
                        self.leadtime_sd = 0
                        return
                    except:
                        self.leadtime = 2
                        self.leadtime_sd = 0
                        return
                else:
                    self.leadtime = 2
                    self.leadtime_sd = 0
                    return

            if self.is_repairable:
                self.procurement_type = 'Repairable'

                # Override
                if self.domain in df_lt_repair.index:
                    try:
                        new_leadtime = df_lt_repair.loc[self.domain]['leadtime_reparable']
                        new_leadtime = float(new_leadtime)

                        self.leadtime_override = True
                        self.leadtime = new_leadtime
                        self.leadtime_sd = 0
                        return
                    except:
                        self.leadtime = 30
                        self.leadtime_sd = 0
                        return
                else:
                    self.leadtime = 30
                    self.leadtime_sd = 0
                    return

            if self.is_buyable:
                self.procurement_type = 'Buyable'
                try:
                    self.leadtime = max([material.leadtime for material in self.materials if material.is_buyable])
                    self.leadtime_sd = max([material.leadtime_sd for material in self.materials if material.is_buyable])
                    return
                except:
                    self.leadtime = 90
                    self.leadtime_sd = 0

            # Else, return Dismountable
            self.procurement_type = 'Dismountable'
            self.leadtime = 90
            self.leadtime_sd = 0
            return
        else:
            raise Exception("procurement_mode not valid.")

    def set_criticality(self):
        list_aux = [material.criticality for material in self.materials]
        if 'critico' in list_aux:
            self.criticality = 'critico'
        elif 'mayor' in list_aux:
            self.criticality = 'mayor'
        elif 'bajo' in list_aux:
            self.criticality = 'bajo'
        else:
            self.has_criticality_data = False
            self.criticality = 'bajo'

    def set_connection_strength(self):
        # If the GEU has more than 1 material, check completion
        if len(self.graph.nodes) > 1:
            self.is_strongly_connected = nx.is_strongly_connected(self.graph)
        else:
            self.is_strongly_connected = True

    def plot_bare_graph(self, show_plot=True, clf: bool = True):
        """
        Plots NetworkX GEU graph with its appropriate color
        :param show_plot: bool
        :param clf: bool
        :return: Void
        """

        if clf:
            for i in plt.get_fignums():
                if plt.figure(i).get_label()[0:5] == "(NXG)":
                    plt.close(plt.figure(i).get_label())
        # Close plot with the same name as the one we're creating (if applies)
        for i in plt.get_fignums():
            if plt.figure(i).get_label() == f"(NXG) GEU {self.catalog}":
                plt.close(f"(NXG) GEU {self.catalog}")
        # Create plot
        plt.figure(f"(NXG) GEU {self.catalog}")

        # Set node colors by domain

        domain_palette = ['#74299E',
                          '#235785',
                          '#7C1F48',
                          '#B48121',
                          '#5D6814',
                          '#0F5A0F',
                          '#818E19',
                          '#1818A8',
                          '#0300A7']
        colors = {'TRANSPORTE - TX': domain_palette[0],
                  'TRANSPORTE - DX': domain_palette[1],
                  'TX - RADIOENLACES Y SATELITAL': domain_palette[2],
                  'ACCESO - FIJA': domain_palette[3],
                  'ACCESO - MOVIL': domain_palette[4],
                  'CORE VOZ': domain_palette[5],
                  'ENTORNO': domain_palette[6],
                  'CMTS': domain_palette[7],
                  'Other': domain_palette[8]}

        # If GEU has many domains, paint each node with its corresponding color
        if self.has_multiple_domains:
            color_map = []
            for node in self.graph.nodes:
                for mat in self.materials:
                    # If it finds a match, use object Material to get node's domain
                    if mat.catalog == node:
                        domain = mat.domain
                        color_map.append(colors[domain])
            color_map_in_use = color_map
        # If that's not the case, the only color is the corresponding one
        else:
            try:
                color_map_in_use = colors[self.domain]
            except:
                color_map_in_use = domain_palette[7]

        # Plot graph
        nx.draw(self.graph, with_labels=True, node_color=color_map_in_use)
        if show_plot:
            plt.show()
        else:
            return plt

    def draw_graph(self, clf: bool, extra_info: bool, path: str, save_png_mode: bool = False):
        """
        Draws NetworkX GEU graph for display or for saving it as PNG
        :param clf: bool
        :param extra_info: bool
        :param path: str
        :param save_png_mode: bool
        :return: Void
        """

        # ----------PREPARATION----------

        # Clear and close fig
        # plt.ioff()
        if clf:
            for i in plt.get_fignums():
                if plt.figure(i).get_label()[0:5] == "(NXG)":
                    plt.close(plt.figure(i).get_label())
        # Close plot with the same name as the one we're creating (if applies)
        for i in plt.get_fignums():
            if plt.figure(i).get_label() == f"(NXG) GEU {self.catalog}":
                plt.close(f"(NXG) GEU {self.catalog}")
        # Create plot
        plt.figure(f"(NXG) GEU {self.catalog}")

        # Set node colors by domain
        domain_palette = ['#74299E',
                          '#235785',
                          '#7C1F48',
                          '#B48121',
                          '#5D6814',
                          '#0F5A0F',
                          '#818E19',
                          '#1818A8',
                          '#0300A7']
        '''domain_palette = ['lightcoral',
                          'bisque',
                          'orange',
                          'cornflowerblue',
                          'deepskyblue',
                          'violet',
                          'lightgrey',
                          'peru',
                          'lime']'''
        colors = {'TRANSPORTE - TX': domain_palette[0],
                  'TRANSPORTE - DX': domain_palette[1],
                  'TX - RADIOENLACES Y SATELITAL': domain_palette[2],
                  'ACCESO - FIJA': domain_palette[3],
                  'ACCESO - MOVIL': domain_palette[4],
                  'CORE VOZ': domain_palette[5],
                  'ENTORNO': domain_palette[6],
                  'CMTS': domain_palette[7],
                  'Other': domain_palette[8]}

        # Extra information
        if extra_info:
            # Adding ax
            plt.gcf().set_figwidth(8)
            plt.gcf().set_figheight(5)
            x_relpos_info = 0.75
            y_relpos_info = 0.35

        # Title
        plt.gcf().suptitle('GEU ' + str(self.catalog) + ': ' + self.name, size=14)

        # Subtitle
        # 1) Graph strength
        subtitle = 'Completo'
        if self.is_strongly_connected is False:
            subtitle = 'Incompleto'
        # 2) Has many domains
        if self.has_multiple_domains:
            subtitle += ', '
            subtitle += 'Múltiples redes'
        plt.gca().set_title(subtitle, size=12)

        # If GEU has many domains, paint each node with its corresponding color
        if self.has_multiple_domains:
            color_map = []
            for node in self.graph.nodes:
                for mat in self.materials:
                    # If it finds a match, use object Material to get node's domain
                    if mat.catalog == node:
                        domain = mat.domain
                        color_map.append(colors[domain])
            color_map_in_use = color_map
        # If that's not the case, the only color is the corresponding one
        else:
            try:
                color_map_in_use = colors[self.domain]
            except:
                color_map_in_use = domain_palette[7]

        # ----------EXTRA INFO----------

        # Extra info
        if extra_info:
            # Making info_bubble
            info_bubble = 'Marca: ' + self.brand + '\n'
            info_bubble += 'Equipo: ' + self.equipment + '\n'
            info_bubble += 'Dominio: ' + '\n'
            info_bubble += self.domain + '\n'
            info_bubble += 'Subarea: ' + '\n'
            info_bubble += self.subarea

            plt.gca().text(x_relpos_info, y_relpos_info, info_bubble, transform=plt.gca().transAxes, fontsize=14)

        # ----------PLOT GRAPH----------

        if self.graph.order() == 1:
            nx.draw_circular(self.graph, with_labels=True, node_color=color_map_in_use)
        elif extra_info:
            # Space a little bit to the left, and then draw
            pos1 = nx.spring_layout(self.graph)

            counter = 0
            for k, v in pos1.items():
                # Shift the x values of every node by 10 to the right
                if self.graph.order() == 2:
                    fixed_x = [0, 0.33]
                    v[0] = fixed_x[counter]
                    counter += 1
                else:
                    v[0] = v[0] - 1.8

            # Add empty anchor graph
            empty_graph = nx.Graph()
            empty_graph.add_node(0)
            nx.draw(empty_graph, node_color='white')

            # Draw graph
            nx.draw(self.graph, pos1, with_labels=True, node_color=color_map_in_use)
        else:
            nx.draw(self.graph, with_labels=True, node_color=color_map_in_use)

        # ----------SHOW OR SAVE PNG----------

        if save_png_mode:
            # Saving PNG files

            relative_path = 'exportables/GEUs'
            if path == '':
                path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', relative_path))
            else:
                if path[-1] != '/':
                    path += '/'
                path += relative_path

            # Save PNG
            plt.savefig(os.path.join(path, 'GEU_' + self.catalog + ".png"), format="PNG")

            # Clear and close fig
            plt.ioff()
            plt.clf()
            plt.close()
        else:
            # Just viewing graph
            plt.show()

    def view_graph(self, clf: bool = True, extra_info: bool = False):
        self.draw_graph(clf=clf, extra_info=extra_info, path='', save_png_mode=False)

    def save_graph_to_png(self, extra_info: bool = False, path: str = ''):
        self.draw_graph(clf=True, extra_info=extra_info, path=path, save_png_mode=True)

    def view_stock(self, path, save_png_mode=False):
        # Shows GEU's historical stock quantities

        # Clear and close fig
        plt.clf()
        # plt.close()

        '''# Extra information
        if extra_info is True:
            # Adding ax
            fig, ax = plt.subplots()
            fig.set_figwidth(8)
            fig.set_figheight(5)
            x_relpos_info = 0.75
            y_relpos_info = 0.40'''

        # Title
        plt.suptitle('Stock histórico del GEU', size=14)

        '''# Subtitle
        # 1) Graph strength
        subtitle = 'Completo'
        if self.is_strongly_connected is False:
            subtitle = 'Incompleto'
        # 2) Has many domains
        if self.domain == 'Many':
            subtitle += ', '
            subtitle += 'Multiples redes'
        plt.title(subtitle, size=12)'''

        '''x_data = ingresos[i][0]['fecha']
        y_data = ingresos[i][j + 1]

        plt.plot(x_data, yd)'''

        plt.suptitle('Ingresos según fecha de corrida', fontsize=24)

        if save_png_mode is True:
            # Saving PNG files
            plt.savefig(path + '/GEU_' + self.catalog + "_stock.png", format="PNG")
        else:
            # Just viewing plot
            plt.show()

    def get_info(self, is_streamlit=False):

        # Catalog (GEU)
        if is_streamlit:
            st.markdown(f'## **GEU {self.catalog} - {self.name}**')
            st.markdown(':package: Materiales:')
            for material in self.materials:
                st.markdown(f'&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp- {str(material.catalog)}: {str(material.name)}')
            st.markdown(f':information_source: Dominio: {self.domain}')
            st.markdown(f':mag_right: Marca: {self.brand}')
            st.markdown(f':briefcase: Peso: {self.weight}')
            st.markdown(f':diamond_shape_with_a_dot_inside: Cluster: {self.cluster}')
            st.markdown(f':chart_with_upwards_trend: Criticidad: {self.criticality}')
            st.markdown(f':heavy_dollar_sign: Precio: {self.price_unit} {self.price}')
            st.markdown(f':heavy_dollar_sign: Reparación: {self.rep_unit} {self.rep_price}')
            st.markdown(f':hourglass: Lead time: {self.leadtime} +/- {self.leadtime_sd} días')
            st.markdown(':truck: Método(s) de abastecimiento:')
            if self.is_spm:
                st.markdown('&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp- SPM')
            if self.is_buyable:
                st.markdown('&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp- Compra')
            if self.is_repairable:
                st.markdown('&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp- Reparación')
            if self.is_dismountable:
                st.markdown('&nbsp &nbsp &nbsp &nbsp &nbsp &nbsp- Desmonte')
        else:
            # Get GEU's info

            catalog_string = '['
            counter = 1

            # Materials
            materials_in_geu = len(self.materials)
            for mat in self.materials:
                catalog_string += mat.catalog
                if counter < materials_in_geu:
                    catalog_string += ', '
                    counter += 1
                else:
                    catalog_string += ']'

            # Catalog (GEU)
            print('GEU', self.catalog + ':', catalog_string)
            # Domain (GEU)
            print('\t' + 'Dominio:', self.domain)
            # Cluster (GEU)
            print('\t' + 'Cluster:', self.cluster)
            # Criticality (GEU)
            print('\t' + 'Criticidad:', self.criticality)
            # Price (GEU)
            print('\t' + f'Precio: US$ {round(self.price, 2)} ({self.price_type})')
            # Lead time (GEU)
            print('\t' + f'Lead time: ({round(self.leadtime, 1)} +/- {round(self.leadtime_sd, 1)}) días')

            # Replenishment method (GEU)
            print('\t' + 'Método(s) de abastecimiento:')
            if not self.has_procurement_type:
                print('\t' + ' -NINGUNO (se imputa Comprable)')
            else:
                if self.is_spm:
                    print('\t' + ' -SPM')
                if self.is_buyable:
                    print('\t' + ' -Compra')
                if self.is_repairable:
                    print('\t' + ' -Reparación')
                if self.is_dismountable:
                    print('\t' + ' -Desmonte')
