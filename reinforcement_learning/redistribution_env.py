import os
import pickle as pkl
import gym
from gym import spaces
from gym.utils import seeding
import datetime as dt
import numpy as np
import math
import pandas as pd
#from pulp import LpVariable, LpMinimize, LpProblem, lpSum, PULP_CBC_CMD

from classes.geu import GEU
from data_loader_class import DataClass
from classes.period import Period
from classes.ticket import Ticket
from classes.stock_in_transit import StockInTransit

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class SimulationEnvironment(gym.Env):

    def __init__(self, env_config={}):

        # Environment Attributes
        self.resupplies_per_year: int = env_config.get('resupplies_per_year', 1)
        self.days_between_movements: int = env_config.get("days_between_movements", 15)
        self.ticket_sla: int = env_config.get("ticket_sla", 2)
        self.ticket_expiration: int = env_config.get("ticket_sla", 7)
        self.get_info = env_config.get("get_info", False)
        self.train = env_config.get("train", False)
        self.start_date: dt.datetime = env_config.get("start_date")
        self.end_date: dt.datetime = env_config.get("end_date")
        self.use_historic_stocks = env_config.get("use_historic_stocks", False)
        self.is_streamlit = env_config.get("is_streamlit", False)
        self.bar_progress = env_config.get("bar_progress", None)
        self.print_on = env_config.get("print_on", True)
        self.multiplicador = env_config.get("multiplicador")
        self.is_s4 = env_config.get("is_s4", False)

        if not self.start_date or not self.end_date:
            raise NotImplemented('Start date or end date are not defined')

        # Data
        self.simulate_purchases = env_config.get("simulate_purchases", True)
        self.grep_type = env_config.get('grep_type', '7_crams')
        self.from_pickle: bool = env_config.get('from_pickle', False)
        if self.from_pickle:
            self.data_class = DataClass.load_from_file(grep_type=self.grep_type)
        else:
            self.data_class = DataClass(print_on=self.print_on,
                                        grep_type=self.grep_type,
                                        is_streamlit=self.is_streamlit,
                                        bar_progress=self.bar_progress,
                                        start_date=self.start_date,
                                        is_s4=self.is_s4)
        self.num_wh = len(self.data_class.wh_list)
        if not self.multiplicador:
            self.multiplicador = self.get_pickled_multiplicador()
        if self.use_historic_stocks:
            print('Using historic stocks')
            self.initial_stocks = self.data_class.df_initial_stocks
            self.initial_stocks = self.initial_stocks[["geu", "fecha", "grep", "stock"]]

        # Environment Spaces
        # # Movements between wh
        self.low_level_action_space = spaces.Box(low=0, high=100,
                                                 shape=(int(self.num_wh / 2 * (self.num_wh - 1)),))
        self.action_space = spaces.Discrete(8)
        # # Stock, Forecast, Unmet demand, Cluster
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(2, 3)), spaces.Discrete(3)))

        # Constants
        self.max_request = env_config.get("max_request", 8)
        self.logistic_divider = env_config.get("logistic_divider", 10)
        self.max_order = env_config.get("max_order", 50)
        self.weight_sla = env_config.get('weight_sla', {'in': 1, 'out': 0.5})
        self.redistribution_divider = env_config.get('redistribution_divider', 1000)

        # Episode Attributes
        self.today: dt.datetime = self.start_date
        self.geu: GEU = None
        self.stock_by_wh = np.array([])
        self.tickets_for_sl: [Ticket] = []
        self.active_tickets_step: [Ticket] = []
        self.inbound_tickets: [Ticket] = []
        self.render_fulfilled_tickets: [Ticket] = []
        self.active_movements: [StockInTransit] = []
        self.step_movements: [StockInTransit] = []
        self.initial_stock_step: [int] = []
        self.ticket_amount_closed_previous_step: int = 0
        self.step_number: int = 0
        self.info = {}
        self.info_tickets = {}
        self.days_between_resupplies: int = None
        self.amount_resupplies: dt.datetime = None

        self.count_ticket_sl: {str: int} = {}

        self.seed()

        if env_config.get('geu_id'):
            self.geu = self.data_class.geus_by_id[env_config['geu_id']]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def is_done(self):
        return self.today >= self.end_date

    def display_init_info(self):

        print("\nInitial Date: " + str(self.today))
        print("Chosen GEU: " + str(self.geu.id))
        print("\nInitial stock: \n")
        self.show_ascii_plot()
        print('\nTotal Stock: ', sum(self.stock_by_wh), "\n")

    def get_reward(self):

        movements_cost = sum([self.data_class.get_moving_price(self.geu.weight * movement.amount, movement.leadtime) for movement in self.step_movements])
        ticket_cost = self.count_ticket_sl['On_time'] * self.weight_sla['in'] +\
                      self.count_ticket_sl['Late'] * self.weight_sla['out'] -\
                      (self.count_ticket_sl['Total'] - self.count_ticket_sl['On_time'] -
                       self.count_ticket_sl['Late'] * self.weight_sla['out'])

        return ticket_cost - movements_cost / self.redistribution_divider

    def get_obs(self):
        # Cluster
        cluster = self.geu.cluster - 1

        # Forecast
        next_step_period = Period(self.today, "d", self.days_between_movements)
        next_step_forecast = np.array([self.geu.get_forecast(next_step_period, wh) for wh in self.data_class.wh_list])
        next_step_forecast = np.array([next_step_forecast[0], np.sum(next_step_forecast[1:])])
        next_step_forecast = (1. / (1. + np.exp(-(next_step_forecast / self.logistic_divider))) - 0.5) * 2

        # Stock
        stock = np.array([self.stock_by_wh[0], np.sum(self.stock_by_wh[1:])])
        stock = (1. / (1. + np.exp(-(stock / self.logistic_divider))) - 0.5) * 2

        # Unmet demand
        unmet_demand = np.zeros(len(self.data_class.wh_list))
        for ticket in self.active_tickets_step:
            unmet_demand[self.data_class.wh_list.index(ticket.grep)] += ticket.amount
        unmet_demand = np.array([unmet_demand[0], np.sum(unmet_demand[1:])])
        unmet_demand = (1. / (1. + np.exp(-(unmet_demand / self.logistic_divider))) - 0.5) * 2

        # Into ray box
        box = np.stack([stock, next_step_forecast, unmet_demand], -1)

        return (box, cluster)

    def reset(self):
        if not self.geu or self.train:
            # Choose random GEU
            self.geu = self.np_random.choice(
                list(filter(lambda x: x.cluster != 0, self.data_class.geus_by_id.values())))

        if self.use_historic_stocks:
            df_stocks = self.initial_stocks.loc[self.initial_stocks['geu'] == self.geu.id, ['grep', 'stock']].copy()
            df_stocks.set_index('grep', inplace=True)
            for wh in self.data_class.wh_list:
                if wh not in df_stocks.index:
                    df_stocks.loc[wh] = 0
            self.stock_by_wh = np.array([df_stocks.loc[wh][0] for wh in self.data_class.wh_list])
        else:
            self.stock_by_wh = np.array([0] * len(self.data_class.wh_list))

        # Tickets
        self.ticket_amount_closed_previous_step = 0
        self.tickets_for_sl = []
        self.active_tickets_step = []
        self.inbound_tickets = []
        self.count_ticket_sl: {str: int} = {'On_time': 0, 'Late': 0, 'Total': 0}
        self.info_tickets = {"Total": 0, 'On_time': 0}

        # Movements
        self.active_movements = []
        self.step_movements = []

        # Resupplies
        self.days_between_resupplies = int(365/self.resupplies_per_year)

        self.today = self.start_date
        self.amount_resupplies = 0

        self.step_number = 0

        # Get Info
        if self.get_info:
            self.info = {"Stock": {self.today: self.stock_by_wh.copy()},
                         "On_time": {self.today: [0] * self.num_wh},
                         "Total": {self.today: [0] * self.num_wh},
                         "Mlen": {self.today: 0},
                         "Mtime": {self.today: 0},
                         "Mttime": {self.today: 0},
                         'mov_cost': {self.today: 0},
                         "Compras": {}}

        return self.get_obs()

    def step(self, action: spaces.Discrete):
        if not self.action_space.contains(action):
            print("Assertion Error!")
            print(f"Action space: {self.action_space}")
            print(f"Action: {action}")
        assert self.action_space.contains(action)

        if self.simulate_purchases and (self.today - self.start_date).days >= self.days_between_resupplies * self.amount_resupplies:
            self.sim_purchases(get_purchase=False)
            self.amount_resupplies += 1

        self.step_movements = []
        self.render_fulfilled_tickets = []

        self.initial_stock_step = list(self.stock_by_wh)

        # Set tickets for service level step
        forecast_period = Period(self.today + dt.timedelta(days=1), "d", self.days_between_movements)
        [self.inbound_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
         if forecast_period.contains(day)]
        self.inbound_tickets = sorted(self.inbound_tickets, key=lambda x: x.ticket_date)

        if self.get_info:
            self.info_tickets['Total'] = [sum([1 for y in self.inbound_tickets if y.grep == x])
                                          for x in self.data_class.wh_list]
            self.info_tickets['On_time'] = [0] * self.num_wh

        self.count_ticket_sl['Total'] = len(self.active_tickets_step) + len(self.inbound_tickets)
        self.count_ticket_sl['On_time'] = 0
        self.count_ticket_sl['Late'] = 0

        # 0. Redistribution Movements
        if self.grep_type != 'unicram':
            self.apply_action(action)

        self.active_movements = sorted(self.active_movements, key=lambda x: x.arrival_date)

        for day in range(self.days_between_movements):
            self.today += dt.timedelta(days=1)

            # 1. Inbound Movements
            self.update_inbound_movements()

            # 2. Update Inbound Tickets
            self.update_inbound_tickets()

            if self.get_info:
                self.info["Stock"][self.today] = self.stock_by_wh.copy()

        self.geu.update_demand_priorities(self.data_class.wh_list, self.today)

        if self.get_info:
            self.info["On_time"][self.today] = self.info_tickets['On_time']
            self.info["Total"][self.today] = self.info_tickets['Total']
            self.info["Mlen"][self.today] = len(self.step_movements)
            self.info["Mtime"][self.today] = sum([movement.leadtime for movement in self.step_movements])
            self.info["Mttime"][self.today] = sum([movement.leadtime * movement.amount for movement in self.step_movements])
            self.info['mov_cost'][self.today] = sum([self.data_class.get_moving_price(weight=self.geu.weight * movement.amount,
                                                                                      travel_time=movement.leadtime)
                                                      for movement in self.step_movements])
 
        self.step_number += 1
        # self.render()
        return self.get_obs(), self.get_reward(), self.is_done(), self.info

    def render(self, mode='human'):

        # General Data
        print('\nStep', self.step_number, '\tGEU:', self.geu.id)
        print("\nDate: " + str(self.today))
        print("Simulated step: " + str(self.days_between_movements) + " days")

        # Transit Stock
        print("\nStocks in transit:\n")
        transit_stock_df = pd.DataFrame(index=self.data_class.wh_list, columns=self.data_class.wh_list)
        transit_stock_df = transit_stock_df.fillna(0)
        transit_stock_df.index.name = "Donating Wh"
        for transit_stock in self.step_movements:
            transit_stock_df.at[transit_stock.donating_wh,
                                transit_stock.receiving_wh] = transit_stock_df.at[transit_stock.donating_wh,
                                                                                  transit_stock.receiving_wh] \
                                                              + transit_stock.amount
        print(transit_stock_df)

        # Tickets
        print("\nActive tickets:")

        grep, amount, date, amount_left = [], [], [], []
        for ticket in self.active_tickets_step:
            grep.append(ticket.grep)
            amount.append(ticket.amount)
            amount_left.append(ticket.amount_left)
            date.append(ticket.ticket_date)
        print(pd.DataFrame({'Grep': grep, 'Cantidad': amount, 'Final': amount_left, 'Fecha': date}))

        # Tickets
        print("\nFulfilled tickets:")

        grep, amount, date, amount_left = [], [], [], []
        for ticket in self.render_fulfilled_tickets:
            grep.append(ticket.grep)
            amount.append(ticket.amount)
            amount_left.append(ticket.amount_left)
            date.append(ticket.ticket_date)
        print(pd.DataFrame({'Grep': grep, 'Cantidad': amount, 'Final': amount_left, 'Fecha': date}))

        print("\nTickets: ", self.count_ticket_sl)

        # Reward
        print("\nReward: " + str(self.get_reward()))

        # Final Stock
        print("\nStocks:\n")
        self.show_ascii_plot()
        print("Total Stock: ", sum(self.stock_by_wh))

        # self.test_validation(type="step")

        print("\n-----------------------------------------------------------\n")

    def show_ascii_plot(self):
        max_value = max(self.stock_by_wh)
        if max_value > 0:
            increment = max_value / 25
            longest_label_length = max(len(label) for label in self.data_class.wh_list)
            for index in range(len(self.stock_by_wh)):
                bar_chunks, remainder = divmod(int(self.stock_by_wh[index] * 8 / increment), 8)
                bar = '█' * bar_chunks
                if remainder > 0:
                    bar += chr(ord('█') + (8 - remainder))
                bar = bar or '▏'
                print(
                    f'{self.data_class.wh_list[index].rjust(longest_label_length)} ▏ {int(self.stock_by_wh[index]):#4d} {bar}')

    def update_inbound_movements(self):
        has_inbound = False

        while self.active_movements and (self.active_movements[0].arrival_date - self.today).days == 0:
            active_movement = self.active_movements.pop(0)
            receiving_wh_index = self.data_class.wh_list.index(active_movement.receiving_wh)
            self.stock_by_wh[receiving_wh_index] = self.stock_by_wh[receiving_wh_index] + active_movement.amount
            has_inbound = True

        if has_inbound and self.active_tickets_step:
            self.update_active_tickets()

    def update_active_tickets(self):
        i = 0
        while i < len(self.active_tickets_step):
            ticket = self.active_tickets_step[i]
            index_ticket_grep = self.data_class.wh_list.index(ticket.grep)
            # if vencido remove
            if (self.today - ticket.ticket_date).days > self.ticket_expiration:
                self.active_tickets_step.pop(i)
                continue
            # if not vencido fullfill
            if self.stock_by_wh[index_ticket_grep] >= ticket.amount_left:
                # Fulfills ticket
                self.stock_by_wh[index_ticket_grep] -= ticket.amount_left
                if (self.today - ticket.ticket_date).days > self.ticket_sla:
                    self.count_ticket_sl['Late'] += 1
                    ticket.is_broken = True
                else:
                    self.count_ticket_sl['On_time'] += 1
                    if self.get_info:
                        self.info_tickets['On_time'][int(self.data_class.wh_list.index(ticket.grep))] += 1
                self.render_fulfilled_tickets.append(self.active_tickets_step.pop(i))
            else:
                # If theres stock partially fulfills ticket
                if self.stock_by_wh[index_ticket_grep] > 0:
                    self.active_tickets_step[i].amount_left -= self.stock_by_wh[index_ticket_grep]
                    self.active_tickets_step[i].is_partial_closed = True
                    self.stock_by_wh[index_ticket_grep] = 0
                i += 1

    def update_inbound_tickets(self):
        while self.inbound_tickets and (self.inbound_tickets[0].ticket_date - self.today).days == 0:
            ticket = self.inbound_tickets.pop(0)
            index_ticket_grep = self.data_class.wh_list.index(ticket.grep)
            if self.stock_by_wh[index_ticket_grep] >= ticket.amount:
                # Fulfills ticket
                self.stock_by_wh[index_ticket_grep] -= ticket.amount
                self.count_ticket_sl['On_time'] += 1
                if self.get_info:
                    self.info_tickets['On_time'][int(self.data_class.wh_list.index(ticket.grep))] += 1
                self.render_fulfilled_tickets.append(ticket)
            else:
                if self.stock_by_wh[index_ticket_grep] > 0:
                    # Partially Fulfills ticket
                    ticket.amount_left -= self.stock_by_wh[index_ticket_grep]
                    ticket.is_partial_closed = True
                    self.stock_by_wh[index_ticket_grep] = 0
                self.active_tickets_step.append(ticket)

    def sim_purchases(self, get_purchase=False):
        forecast_window_days = 365
        #forecast_window_month = 6
        next_step_period = Period(self.today, "d", forecast_window_days)
        days_stock_purchase = self.days_between_resupplies + self.geu.leadtime
        next_step_forecast = self.geu.get_forecast(next_step_period, "all", type='expo') * days_stock_purchase / forecast_window_days

        teoric_stock = sum(self.stock_by_wh) + sum([mov.amount for mov in self.active_movements])

        ask = max(0, next_step_forecast * self.multiplicador[self.geu.domain][self.geu.cluster - 1][0] - teoric_stock +
                  self.multiplicador[self.geu.domain][self.geu.cluster - 1][1])

        if get_purchase:
            return round(ask,0), self.geu.leadtime, round(next_step_forecast,2)
        else:
            self.info["Compras"][self.today] = round(ask, 0)
            if round(ask) > 0:
                self.purchase_items([round(ask)], self.geu.leadtime)

    def sim_redistribute_materials(self, action):
        # tuple (i, j)
        # i: donating warehouse
        # j: receiving warehouse
        # index_count : action numpy index
        # action is a numpy array

        index_count = 0

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                valor = action[index_count]

                if valor > 0 and self.stock_by_wh[i] > 0:
                    donation = min(self.stock_by_wh[i], valor)
                    if donation > 0:
                        self.stock_by_wh[i] -= donation

                        leadtime = self.data_class.leadtime_between_warehouses.get(
                            (self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j]), 4)
                        if self.step_number == 0:
                            leadtime = 1

                        object_movement = StockInTransit(receiving_wh=self.data_class.wh_dictionary[j],
                                                         date=self.today,
                                                         amount=donation,
                                                         leadtime=leadtime,
                                                         donating_wh=self.data_class.wh_dictionary[i])
                        self.active_movements.append(object_movement)
                        self.step_movements.append(object_movement)
                elif valor < 0 and self.stock_by_wh[j] > 0:
                    donation = min(self.stock_by_wh[j], - valor)
                    if donation > 0:
                        self.stock_by_wh[j] -= donation
                        leadtime = self.data_class.leadtime_between_warehouses.get(
                            (self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i]), 4)
                        if self.step_number == 0:
                            leadtime = 1
                        object_movement = StockInTransit(receiving_wh=self.data_class.wh_dictionary[i],
                                                         date=self.today,
                                                         amount=donation,
                                                         leadtime=leadtime,
                                                         donating_wh=self.data_class.wh_dictionary[j])
                        self.active_movements.append(object_movement)
                        self.step_movements.append(object_movement)

                index_count += 1

    def test_validation(self, type: str):
        if type == "step":
            integer_stock = all([isinstance(stock, int) for stock in self.stock_by_wh])
            non_negative_stock = all([(stock >= 0) for stock in self.stock_by_wh])

            initial_stock = sum(self.initial_stock_step)
            final_stock = sum(self.stock_by_wh)
            transit_stock = sum([transit_stock.amount for transit_stock in self.active_movements])
            tickets_amount_outflow = sum([(ticket.amount - ticket.amount_left) for ticket in self.tickets_for_sl
                                          if ticket.is_closed or ticket.is_partial_closed])

            balance_system = (final_stock == initial_stock + transit_stock +
                              (-tickets_amount_outflow + self.ticket_amount_closed_previous_step))

            self.ticket_amount_closed_previous_step = tickets_amount_outflow

            print('\nValidation:\n')
            print('\tInteger stock', (20 - len("Integer stock")) * " ",
                  "✓ [Checked]" if integer_stock else "⚠ [Warning]")
            print('\tNon negative stock', (20 - len("Non negative stock")) * " ",
                  "✓ [Checked]" if non_negative_stock else "⚠ [Warning]")
            print('\tBalanced system', (20 - len("Balanced system")) * " ",
                  "✓ [Checked]" if balance_system else "⚠ [Warning]")

    def purchase_items(self, action, lt):
        if self.step_number == 0:
            self.stock_by_wh[0] += round(action[0])
        else:
            object_movement = StockInTransit(receiving_wh=self.data_class.wh_list[0],
                                             date=self.today,
                                             amount=round(action[0]),
                                             leadtime=lt)
            self.active_movements.append(object_movement)

    def get_pickled_multiplicador(self):
        filename = os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{self.grep_type}_{self.resupplies_per_year}_low.pkl')
        if os.path.isfile(filename):
            with open(filename, 'rb') as file:
                multiplicador = pkl.load(file)
        else:
            print('Using default multiplier')
            multiplicador = {domain: [(1, 2)] * len(DataClass.clusters) for domain in self.data_class.domain_list}
        return multiplicador


    ''' High Level Actions
            Input: Nada
            Output: Vector de accion
    '''

    def action1(self):
        """1. Level out stock, minimizing leadtime"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        #
        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        # Calculate balance stock
        stock_by_wh = np.copy(self.stock_by_wh)
        balance_stock = math.floor(np.sum(stock_by_wh) / stock_by_wh.size)
        #
        if balance_stock > 0:
            excess_stock = stock_by_wh - balance_stock
        else:
            excess_stock = stock_by_wh - 1
        #
        # Loop while excess
        while np.amax(excess_stock) != 0 and np.amin(excess_stock) < 0:
        #     # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation

        # ---------------------------------------------------------------------------------------------------------------

        """
        stock_dict = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(self.stock_by_wh)}
        wh_list = self.data_class.wh_list

        leadtimes = self.data_class.df_leadtimes

        # Define model--------------------------------------------------------------------------------------------------
        model = LpProblem("Action1", sense=LpMinimize)

        # Define Variables----------------------------------------------------------------------------------------------
        # Initial stock amount
        W0 = LpVariable.dicts('#_wh_initial', [w for w in wh_list], lowBound=0, cat="Integer")

        # Final stock amount
        Wf = LpVariable.dicts('#_wh_final', [w for w in wh_list], lowBound=0, cat="Integer")

        # Amount transportated between warehouese
        X = LpVariable.dicts('movements_warehouse', [(w, w2) for w in wh_list for w2 in wh_list if w2 != w], lowBound=0,
                             cat="Integer")

        # Binaria que se activa si se realiza una distribución de i a j
        I = LpVariable.dicts('I', [(i, j) for i in wh_list for j in wh_list if j != i], cat="Integer")

        M = 1000

        balance_stock = math.floor(sum(self.stock_by_wh) / len(wh_list))

        # Defining Objective--------------------------------------------------------------------------------------------
        model += lpSum([I[(i, j)] * leadtimes.at[i, j] for i in wh_list for j in wh_list if j != i])

        # Constraints---------------------------------------------------------------------------------------------------
        for i in wh_list:
            # Initial Stock
            model += W0[i] == stock_dict[i]

            # Balance stock
            model += Wf[i] >= balance_stock

            # Balance outflows
            model += lpSum([X[(i, j)] for j in wh_list if j != i]) <= W0[i]

            # Balance warehouse
            model += W0[i] + lpSum([X[(j, i)] for j in wh_list if j != i]) == Wf[i] + lpSum(
                [X[(i, j)] for j in wh_list if j != i])

            # Binary constraints
            for j in wh_list:
                if i != j:
                    model += X[(i, j)] <= M * I[(i, j)]


        # Results-------------------------------------------------------------------------------------------------------
        # Solve Model
        model.solve(PULP_CBC_CMD(msg = False))

        # Translate results
        action = []

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                from_i_to_j = X[(self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j])].varValue
                from_j_to_i = X[(self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i])].varValue
                if from_i_to_j != 0:
                    action.append(from_i_to_j)
                elif from_j_to_i != 0:
                    action.append(-from_j_to_i)
                else:
                    action.append(0)
                    
        action = np.array(action)
        """
        return action

    def action1_st(self, grep_availables):
        """1. Level out stock, minimizing leadtime between warehouses enables"""
        # grep_availables is a list of greps availables index of stock by wh

        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Calculate balance stock
        stock_by_wh = np.copy(self.stock_by_wh)
        balance_stock = math.floor(np.sum(stock_by_wh) / grep_availables.size)

        excess_stock = np.copy(stock_by_wh)

        for index in grep_availables:
            if balance_stock > 0:
                excess_stock[index] = stock_by_wh[index] - balance_stock
            else:
                excess_stock[index] = stock_by_wh[index] - 1

        # Loop while excess
        while np.amax(excess_stock) != 0 and np.amin(excess_stock) < 0:
        #     # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = 1
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation

        return action

    def action2(self):
        """2. Compare stock to previous fortnight forecast"""
        # If there is no stock then do nothing
        if sum(self.stock_by_wh) == 0:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Initiate variables
        forecast_by_wh = None
        transit_stock = np.array([0] * len(self.data_class.wh_list))
        is_done = False
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        stock_by_wh = np.copy(self.stock_by_wh)

        # Iterate until no one needs or can give stock
        while not is_done:
            # Calculate stock in transit for warehouse and between redistribution days
            if forecast_by_wh is None:
                next_step_period = Period(self.today, "d", self.days_between_movements)
                forecast_by_wh = np.array([self.geu.get_forecast(next_step_period, wh)
                                           for wh in self.data_class.wh_list])

            # Calculate forecasted stock error
            daily_consumption = forecast_by_wh / self.days_between_movements
            fcst_stock_error = (stock_by_wh + transit_stock) - forecast_by_wh

            # If there's no consumption you've got a virtually infinite amount of days of stock
            daily_consumption = np.where(daily_consumption == 0, 99, daily_consumption)

            # Req/exc stock weighted by consumption (days of stock)
            days_of_stock_by_wh = fcst_stock_error / daily_consumption

            # Get wh more days of stock required by the end of the period (negative number)
            receiving_wh = np.where(days_of_stock_by_wh == np.amin(days_of_stock_by_wh))[0][0]

            if fcst_stock_error[receiving_wh] < 0:
                # List of wh with remaining stock -> available to donate
                valid_wh_donation = np.where(fcst_stock_error > 0)[0]

                # Select wh from available list with the most days of stock remaining by the end of period
                max_aux = 0
                donating_wh = None
                for index in valid_wh_donation:
                    if days_of_stock_by_wh[index] > max_aux and stock_by_wh[index] > 0:
                        max_aux = days_of_stock_by_wh[index]
                        donating_wh = index

                if donating_wh is not None:
                    # Remove stock from donating warehouse
                    stock_by_wh[donating_wh] -= 1
                    transit_stock[receiving_wh] += 1

                    # Add movement to action
                    index = self.get_action_index(wh_origin=donating_wh, wh_destination=receiving_wh)
                    if donating_wh < receiving_wh:
                        action[index] += 1
                    else:
                        action[index] -= 1

                else:
                    is_done = True
            else:
                is_done = True

        return action

    def action3(self):
        """3. Compare stock with forecast until next stock purchase"""
        # If there is no stock then do nothing
        if sum(self.stock_by_wh) == 0:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Initiate variables
        forecast_by_wh = None
        transit_stock = np.array([0] * len(self.data_class.wh_list))
        is_done = False
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        stock_by_wh = np.copy(self.stock_by_wh)

        # Iterate until no one needs or can give stock
        while not is_done:
            # Calculate stock in transit for warehouse and between redistribution days
            if forecast_by_wh is None:
                next_step_period = Period(self.today, "d", self.days_between_resupplies)
                forecast_by_wh = np.array(
                    [self.geu.get_forecast(next_step_period, wh) for wh in self.data_class.wh_list])

            # Calculate forecasted stock error
            daily_consumption = forecast_by_wh / self.days_between_resupplies
            fcst_stock_error = (stock_by_wh + transit_stock) - forecast_by_wh

            # If there's no consumption you've got a virtually infinite amount of days of stock
            daily_consumption = np.where(daily_consumption == 0, 99, daily_consumption)

            # Req/exc stock weighted by consumption (days of stock)
            days_of_stock_by_wh = fcst_stock_error / daily_consumption

            # Get wh more days of stock required by the end of the period (negative number)
            receiving_wh = np.where(days_of_stock_by_wh == np.amin(days_of_stock_by_wh))[0][0]

            if fcst_stock_error[receiving_wh] < 0:
                # List of wh with remaining stock -> available to donate
                valid_wh_donation = np.where(fcst_stock_error > 0)[0]

                # Select wh from available list with the most days of stock remaining by the end of period
                max_aux = 0
                donating_wh = None
                for index in valid_wh_donation:
                    if days_of_stock_by_wh[index] > max_aux and stock_by_wh[index] > 0:
                        max_aux = days_of_stock_by_wh[index]
                        donating_wh = index

                if donating_wh is not None:
                    # Remove stock from donating warehouse
                    stock_by_wh[donating_wh] -= 1
                    transit_stock[receiving_wh] += 1

                    # Add movement to action
                    index = self.get_action_index(wh_origin=donating_wh, wh_destination=receiving_wh)
                    if donating_wh < receiving_wh:
                        action[index] += 1
                    else:
                        action[index] -= 1

                else:
                    is_done = True
            else:
                is_done = True

        return action

    def action4(self):
        """
        4 Concentrate at least 20% of stock in the Stock Central Hub
        Send material to/from closer WH
        """
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock then do nothing
        if sum(self.stock_by_wh) == 0:
            return np.array(action)

        # Get corresponding variables
        bs_as_index = self.data_class.wh_list.index('00')
        stock_by_wh = np.copy(self.stock_by_wh)

        # Calculate 20% of total stock
        bs_as_goal = math.ceil(sum(stock_by_wh) * 0.2)

        # If goal is already satisfied then do nothing
        if stock_by_wh[bs_as_index] >= bs_as_goal:
            return action

        # Get closer WH with stock
        wh_with_stock = np.delete(np.where(stock_by_wh > 0)[0], bs_as_index)
        wh_leadtimes = {wh: self.data_class.leadtime_between_warehouses.get((wh, '00'), 4)
                        for wh in wh_with_stock}

        # Loop to get action
        is_done = len(wh_leadtimes) == 0
        while not is_done:
            # Get next closest WH
            origin = min(wh_leadtimes, key=wh_leadtimes.get)
            origin_index = self.data_class.wh_list.index('0' + str(origin))

            if stock_by_wh[origin_index] > 0:
                # Empty origin WH accordingly
                if origin_index < bs_as_index:
                    action[self.get_action_index(origin_index, bs_as_index)] += stock_by_wh[origin_index]
                else:
                    action[self.get_action_index(origin_index, bs_as_index)] -= stock_by_wh[origin_index]
                stock_by_wh[origin_index] = 0

                # Add this new stock to the Stock Central Hub
                stock_by_wh[bs_as_index] += stock_by_wh[origin_index]

            # Drop this WH from the list
            wh_leadtimes.pop(origin)

            # Stop when goal is achieved or when there are no more WH in the looped list
            if stock_by_wh[bs_as_index] >= bs_as_goal or not bool(wh_leadtimes):
                is_done = True

        return np.array(action)

    def action5(self):
        """5. No movements"""
        return np.array([0] * self.low_level_action_space.shape[0])

    def action6(self):
        """6. Random movements"""
        return self.low_level_action_space.sample()

    def generic_action(self, weight_by: str):
        """Level out stock, weighting by 180-day rolling sum of demand, minimizing leadtime"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        period = Period(self.today - dt.timedelta(days=self.days_between_resupplies),
                        "d", self.days_between_resupplies)

        if weight_by == 'demand':  # Action 7
            # Get demand and stock by wh
            demands_by_wh = np.array(
                [self.geu.get_demand(period, wh) for wh in self.data_class.wh_list])
            stock_by_wh = np.copy(self.stock_by_wh)

            # Calculate total initial stock and demand
            total_initial_stock = np.sum(stock_by_wh)
            total_demand = np.sum(demands_by_wh)

            # If there was no demand then do nothing
            if total_demand == 0:
                return action

            # Weight balance stock according to wh demand
            weights = demands_by_wh / total_demand
            weighted_balance_stock = np.floor(weights * total_initial_stock)
            excess_stock = stock_by_wh - weighted_balance_stock
        elif weight_by == 'tickets':  # Action 9
            list_of_tickets = []
            [list_of_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
             if period.contains(day)]

            ticket_by_wh = np.array(
                [sum([ticket.amount for ticket in list_of_tickets if ticket.grep == wh]) for wh in
                 self.data_class.wh_list])

            stock_by_wh = np.copy(self.stock_by_wh)

            total_initial_stock = np.sum(stock_by_wh)
            total_tickets = np.sum(ticket_by_wh)

            if total_tickets == 0:
                return action

            weights = ticket_by_wh / total_tickets
            weighted_stock = np.floor(weights * total_initial_stock)
            excess_stock = stock_by_wh - weighted_stock
        elif weight_by == 'tickets_lt':  # Action 10
            list_of_tickets = []
            [list_of_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
             if period.contains(day)]

            ticket_by_wh = np.array(
                [sum([ticket.amount for ticket in list_of_tickets if ticket.grep == wh]) for wh in
                 self.data_class.wh_list])

            stock_by_wh = np.copy(self.stock_by_wh)

            total_initial_stock = np.sum(stock_by_wh)
            total_tickets = np.sum(ticket_by_wh)

            if total_tickets == 0:
                return action

            weights = ticket_by_wh / total_tickets
            weighted_stock = np.floor(weights * total_initial_stock)
            excess_stock = stock_by_wh - weighted_stock
        else:
            raise Exception('No valido')

        while np.amax(excess_stock) != 0 and np.amin(excess_stock) < 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation
        return action

    def action7(self):
        """7. Level out stock, weighting by 180-day rolling sum of demand, minimizing leadtime"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        next_step_period = Period(self.today - dt.timedelta(days=self.days_between_resupplies),
                                  "d", self.days_between_resupplies)

        # Get demand and stock by wh
        demands_by_wh = np.array(
            [self.geu.get_demand(next_step_period, wh) for wh in self.data_class.wh_list])
        stock_by_wh = np.copy(self.stock_by_wh)

        # Calculate total initial stock and demand
        total_initial_stock = np.sum(stock_by_wh)
        total_demand = np.sum(demands_by_wh)

        # If there was no demand then do nothing
        if total_demand == 0:
            return action

        # Weight balance stock according to wh demand
        weights = demands_by_wh / total_demand
        weighted_balance_stock = np.floor(weights * total_initial_stock)
        excess_stock = stock_by_wh - weighted_balance_stock

        while np.amax(excess_stock) != 0 and np.amin(excess_stock) < 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation

        """
        # Variables-----------------------------------------------------------------------------------------------------
        next_step_period = Period(self.today - dt.timedelta(days=self.days_between_purchases),
                                  "d",
                                  self.days_between_purchases)
        demands_by_wh = np.array(
            [self.geu.get_demand(next_step_period, wh) for wh in self.data_class.wh_list])

        stocks = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(self.stock_by_wh)}
        demandas = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(demands_by_wh)}
        wh_list = self.data_class.wh_list
        leadtimes = self.data_class.df_leadtimes

        total_initial_stock = sum(list(stocks.values()))
        total_demand = sum(list(demandas.values()))
        if total_demand == 0:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        ponderate = {wh: demandas[wh] / total_demand for wh in wh_list}
        stock_ponderate = {wh: math.floor(ponderate[wh] * total_initial_stock) for wh in wh_list}

        # Initialize model----------------------------------------------------------------------------------------------
        model_7 = LpProblem("Action7", sense=LpMinimize)

        # Model Variables-----------------------------------------------------------------------------------------------
        # Amount moved from werehouse to customer
        W0 = LpVariable.dicts('#_wh_initial', [w for w in wh_list], lowBound=0, cat="Integer")

        Wf = LpVariable.dicts('#_wh_final', [w for w in wh_list], lowBound=0, cat="Integer")

        # Amount moved between werehouses
        X = LpVariable.dicts('movements_warehouse', [(w, w2) for w in wh_list for w2 in wh_list if w2 != w], lowBound=0,
                             cat="Integer")

        # Binary var: marks if a redistribution is made from i to j
        I = LpVariable.dicts('I', [(i, j) for i in wh_list for j in wh_list if j != i], cat="Integer")

        M = 1000

        # Defining Objective--------------------------------------------------------------------------------------------
        model_7 += lpSum([I[(i, j)] * leadtimes.at[i, j] for i in wh_list for j in wh_list if j != i])

        # Constraints---------------------------------------------------------------------------------------------------

        for i in wh_list:
            # Initial Stock
            model_7 += W0[i] == stocks[i]

            # Balance outflows
            model_7 += lpSum([X[(i, j)] for j in wh_list if j != i]) <= W0[i]

            # Balance warehouse
            model_7 += W0[i] + lpSum([X[(j, i)] for j in wh_list if j != i]) == Wf[i] + lpSum(
                [X[(i, j)] for j in wh_list if j != i])

            # Binary constraints
            for j in wh_list:
                if i != j:
                    model_7 += X[(i, j)] <= M * I[(i, j)]

            # Weigh in Stock
            model_7 += Wf[i] >= stock_ponderate[i]

        # Balance Initial & Final Stock
        model_7 += lpSum([W0[i] for i in wh_list]) == lpSum([Wf[i] for i in wh_list])

        # Results-------------------------------------------------------------------------------------------------------
        # Solve Model
        model_7.solve(PULP_CBC_CMD(msg=False))

        # Translate results
        action = []

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                from_i_to_j = X[(self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j])].varValue
                from_j_to_i = X[(self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i])].varValue
                if from_i_to_j != 0:
                    action.append(from_i_to_j)
                elif from_j_to_i != 0:
                    action.append(-from_j_to_i)
                else:
                    action.append(0)

        return np.array(action)
        """
        return action

    '''def action8(self):
        """8. Level out stock, minimizing total shipments"""
        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Get corresponding variables
        stock_dict = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(self.stock_by_wh)}
        wh_list = self.data_class.wh_list

        # Define model--------------------------------------------------------------------------------------------------
        model_8 = LpProblem("Action1", sense=LpMinimize)

        # Define Variables----------------------------------------------------------------------------------------------
        # Initial stock amount
        W0 = LpVariable.dicts('#_wh_initial', [w for w in wh_list], lowBound=0, cat="Integer")

        # Final stock amount
        Wf = LpVariable.dicts('#_wh_final', [w for w in wh_list], lowBound=0, cat="Integer")

        # Amount moved between warehouses
        X = LpVariable.dicts('movements_warehouse', [(w, w2) for w in wh_list for w2 in wh_list if w2 != w], lowBound=0,
                             cat="Integer")

        # Binary var: marks if a redistribution is made from i to j
        I = LpVariable.dicts('I', [(i, j) for i in wh_list for j in wh_list if j != i], cat="Integer")

        M = 1000

        balance_stock = math.floor(sum(self.stock_by_wh) / len(wh_list))

        # Defining Objective--------------------------------------------------------------------------------------------
        model_8 += lpSum([I[(i, j)] for i in wh_list for j in wh_list if j != i])

        # Constraints---------------------------------------------------------------------------------------------------
        for i in wh_list:
            # Initial Stock
            model_8 += W0[i] == stock_dict[i]

            # Balance stock
            model_8 += Wf[i] >= balance_stock

            # Balance outflows
            model_8 += lpSum([X[(i, j)] for j in wh_list if j != i]) <= W0[i]

            # Balance warehouse
            model_8 += W0[i] + lpSum([X[(j, i)] for j in wh_list if j != i]) == Wf[i] + lpSum(
                [X[(i, j)] for j in wh_list if j != i])

            # Binary constraints
            for j in wh_list:
                if i != j:
                    model_8 += X[(i, j)] <= M * I[(i, j)]

        # Results-------------------------------------------------------------------------------------------------------
        # Solve Model
        model_8.solve(PULP_CBC_CMD(msg=False))

        # Translate results
        action = []

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                from_i_to_j = X[(self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j])].varValue
                from_j_to_i = X[(self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i])].varValue
                if from_i_to_j != 0:
                    action.append(from_i_to_j)
                elif from_j_to_i != 0:
                    action.append(-from_j_to_i)
                else:
                    action.append(0)

        return np.array(action)

    def action9(self):
        """9. Level out stock, weighting by 180-day rolling sum of demand, minimizing shipments"""

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Variables-----------------------------------------------------------------------------------------------------
        next_step_period = Period(self.today - dt.timedelta(days=self.days_between_purchases),
                                  "d", self.days_between_purchases)
        demands_by_wh = np.array(
            [self.geu.get_demand(next_step_period, wh) for wh in self.data_class.wh_list])

        stocks = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(self.stock_by_wh)}
        demands = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(demands_by_wh)}
        wh_list = self.data_class.wh_list

        total_initial_stock = sum(list(stocks.values()))
        total_demand = sum(list(demands.values()))
        if total_demand == 0:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        weights = {wh: demands[wh] / total_demand for wh in wh_list}
        weighted_stock = {wh: math.floor(weights[wh] * total_initial_stock) for wh in wh_list}

        # Initialize model----------------------------------------------------------------------------------------------
        model_7 = LpProblem("Action7", sense=LpMinimize)

        # Model Variables-----------------------------------------------------------------------------------------------
        # Amount moved from warehouse to customer
        W0 = LpVariable.dicts('#_wh_initial', [w for w in wh_list], lowBound=0, cat="Integer")

        Wf = LpVariable.dicts('#_wh_final', [w for w in wh_list], lowBound=0, cat="Integer")

        # Amount moved between werehouses
        X = LpVariable.dicts('movements_warehouse', [(w, w2) for w in wh_list for w2 in wh_list if w2 != w],
                             lowBound=0, cat="Integer")

        # Binary var: marks if a redistribution is made from i to j
        I = LpVariable.dicts('I', [(i, j) for i in wh_list for j in wh_list if j != i], cat="Integer")

        M = 1000

        # Defining Objective-------------------------------------------------------------------------------------------
        model_7 += lpSum([I[(i, j)] for i in wh_list for j in wh_list if j != i])

        # Constraints---------------------------------------------------------------------------------------------------

        for i in wh_list:
            # Initial Stock
            model_7 += W0[i] == stocks[i]

            # Balance outflows
            model_7 += lpSum([X[(i, j)] for j in wh_list if j != i]) <= W0[i]

            # Balance warehouse
            model_7 += W0[i] + lpSum([X[(j, i)] for j in wh_list if j != i]) == Wf[i] + lpSum(
                [X[(i, j)] for j in wh_list if j != i])

            # Binary constraints
            for j in wh_list:
                if i != j:
                    model_7 += X[(i, j)] <= M * I[(i, j)]

            # Ponderate Stock
            model_7 += Wf[i] >= weighted_stock[i]

        # Balance Initial & Final Stock
        model_7 += lpSum([W0[i] for i in wh_list]) == lpSum([Wf[i] for i in wh_list])

        # Results-------------------------------------------------------------------------------------------------------
        # Solve Model
        model_7.solve(PULP_CBC_CMD(msg=False))

        # Translate results
        action = []

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                from_i_to_j = X[(self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j])].varValue
                from_j_to_i = X[(self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i])].varValue
                if from_i_to_j != 0:
                    action.append(from_i_to_j)
                elif from_j_to_i != 0:
                    action.append(-from_j_to_i)
                else:
                    action.append(0)

        return np.array(action)'''

    def action10(self):
        """10. Level out stock, weighting by 180-day rolling count of demand tickets, minimizing leadtimes"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        period = Period(self.today - dt.timedelta(days=self.days_between_resupplies),
                        "d", self.days_between_resupplies)

        # Calculate relevant variables
        list_of_tickets = []
        [list_of_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
         if period.contains(day)]
        ticket_by_wh = np.array(
            [sum([ticket.amount for ticket in list_of_tickets if ticket.grep == wh]) for wh in self.data_class.wh_list])
        stock_by_wh = np.copy(self.stock_by_wh)
        total_initial_stock = np.sum(stock_by_wh)
        total_tickets = np.sum(ticket_by_wh)

        if total_tickets == 0:
            return action

        # Calculate excess stock
        weights = ticket_by_wh / total_tickets
        weighted_balance_stock = np.floor(weights * total_initial_stock)
        excess_stock = stock_by_wh - weighted_balance_stock

        while np.amax(excess_stock) != 0 and np.where(excess_stock < 0)[0].size > 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation

        """
        # If non stock then do nothing
        if sum(self.stock_by_wh) < 2:
            return np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # Variables-----------------------------------------------------------------------------------------------------
        period = Period(self.today - dt.timedelta(days=self.days_between_purchases),
                        "d", self.days_between_purchases)
        list_of_tickets = []
        [list_of_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
         if period.contains(day)]

        ticket_by_wh = np.array(
            [sum([ticket.amount for ticket in list_of_tickets if ticket.grep == wh])
             for wh in self.data_class.wh_list])

        stocks = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(self.stock_by_wh)}
        tickets = {self.data_class.wh_dictionary[i]: s for i, s in enumerate(ticket_by_wh)}
        wh_list = self.data_class.wh_list
        leadtimes = self.data_class.df_leadtimes

        # Initialize model----------------------------------------------------------------------------------------------
        model_10 = LpProblem("Action7", sense=LpMinimize)

        # Model Variables-----------------------------------------------------------------------------------------------
        # Amount moved from warehouse to customer
        W0 = LpVariable.dicts('#_wh_initial', [w for w in wh_list], lowBound=0, cat="Integer")

        Wf = LpVariable.dicts('#_wh_final', [w for w in wh_list], lowBound=0, cat="Integer")

        # Amount moved between warehouses
        X = LpVariable.dicts('movements_warehouse', [(w, w2) for w in wh_list for w2 in wh_list if w2 != w], lowBound=0,
                             cat="Integer")

        # Binary var: marks if a redistribution is made from i to j
        I = LpVariable.dicts('I', [(i, j) for i in wh_list for j in wh_list if j != i], cat="Integer")

        M = 1000

        total_initial_stock = sum(list(stocks.values()))
        total_demand = sum(list(tickets.values()))
        ponderate = {wh: tickets[wh] / total_demand for wh in wh_list}
        stock_ponderate = {wh: math.floor(ponderate[wh] * total_initial_stock) for wh in wh_list}

        # Defining Objective--------------------------------------------------------------------------------------------
        model_10 += lpSum([I[(i, j)] * leadtimes.at[i, j] for i in wh_list for j in wh_list if j != i])

        # Constraints---------------------------------------------------------------------------------------------------

        for i in wh_list:
            # Initial Stock
            model_10 += W0[i] == stocks[i]

            # Balance outflows
            model_10 += lpSum([X[(i, j)] for j in wh_list if j != i]) <= W0[i]

            # Balance warehouse
            model_10 += W0[i] + lpSum([X[(j, i)] for j in wh_list if j != i]) == Wf[i] + lpSum(
                [X[(i, j)] for j in wh_list if j != i])

            # Binary constraints
            for j in wh_list:
                if i != j:
                    model_10 += X[(i, j)] <= M * I[(i, j)]

            # Weigh Stock
            model_10 += Wf[i] >= stock_ponderate[i]

        # Balance Initial & Final Stock
        model_10 += lpSum([W0[i] for i in wh_list]) == lpSum([Wf[i] for i in wh_list])

        # Results-------------------------------------------------------------------------------------------------------
        # Solve Model
        model_10.solve(PULP_CBC_CMD(msg = False))

        # Translate results
        action = []

        for i in range(len(self.data_class.wh_list) - 1):
            for j in range(i + 1, len(self.data_class.wh_list)):
                from_i_to_j = X[(self.data_class.wh_dictionary[i], self.data_class.wh_dictionary[j])].varValue
                from_j_to_i = X[(self.data_class.wh_dictionary[j], self.data_class.wh_dictionary[i])].varValue
                if from_i_to_j != 0:
                    action.append(from_i_to_j)
                elif from_j_to_i != 0:
                    action.append(-from_j_to_i)
                else:
                    action.append(0)

        return np.array(action)
        """
        return action

    def action11(self):
        """Level out stock between warehouses that had demands in the last 180 days"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        period = Period(self.today - dt.timedelta(days=self.days_between_resupplies),
                        "d", self.days_between_resupplies)

        # Calculate demands
        demands_by_wh = np.array(
            [self.geu.get_demand(period, wh) for wh in self.data_class.wh_list])

        # If there are no demands then do nothing
        if np.sum(demands_by_wh) == 0:
            return action

        # Count WH with and without tickets
        wh_with_demand = np.where(demands_by_wh > 0)[0]
        wh_with_out_demand = np.where(demands_by_wh <= 0)[0]

        # Calculate balance stock based on WH with tickets
        stock_by_wh = np.copy(self.stock_by_wh)
        balance_stock = math.floor(np.sum(stock_by_wh) / wh_with_demand.size)

        # Calculate excess stock for WH with tickets
        if balance_stock > 0:
            remaining_stock = stock_by_wh - balance_stock
        else:
            remaining_stock = stock_by_wh - 1

        # Set excess stock for WH without tickets as 0
        for wh_index in wh_with_out_demand:
            remaining_stock[wh_index] = 0

        while np.amax(remaining_stock) != 0 and np.where(remaining_stock < 0)[0].size > 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(remaining_stock == np.amin(remaining_stock))[0][0]
            max_pos_index = np.where(remaining_stock == np.amax(remaining_stock))[0][0]

            # Donate accordingly
            amount_donation = min(remaining_stock[max_pos_index], -remaining_stock[max_neg_index])
            remaining_stock[max_pos_index] -= amount_donation
            remaining_stock[max_neg_index] += amount_donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += amount_donation
            else:
                action[index] -= amount_donation
        return action

    def action12(self):
        """Nivelar stock entre los almacenes que hayan tenido tickets 180 días"""
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        # Calculate relevant variables
        period = Period(self.today - dt.timedelta(days=self.days_between_resupplies),
                        "d", self.days_between_resupplies)
        list_of_tickets = []
        [list_of_tickets.extend(self.geu.tickets[day]) for day in self.geu.tickets.keys()
         if period.contains(day)]
        ticket_by_wh = np.array(
            [sum([ticket.amount for ticket in list_of_tickets if ticket.grep == wh]) for wh in self.data_class.wh_list])

        # If there are no tickets then do nothing
        if np.sum(ticket_by_wh) == 0:
            return action

        # Count WH with and without tickets
        wh_with_tickets = np.where(ticket_by_wh > 0)[0]
        wh_with_out_tickets = np.where(ticket_by_wh <= 0)[0]

        # Calculate balance stock based on WH with tickets
        stock_by_wh = np.copy(self.stock_by_wh)
        balance_stock = math.floor(np.sum(stock_by_wh) / wh_with_tickets.size)

        # Calculate excess stock for WH with tickets
        if balance_stock > 0:
            excess_stock = stock_by_wh - balance_stock
        else:
            excess_stock = stock_by_wh - 1

        # Set excess stock for WH without tickets as 0
        for wh_index in wh_with_out_tickets:
            excess_stock[wh_index] = 0

        while np.amax(excess_stock) != 0 and np.where(excess_stock < 0)[0].size > 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation
        return action

    def action13(self):
        """Balance with minimum of 1 o stock per wh and the rest compare with forecast"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))
        # If balancing is not possible then do nothing
        if sum(self.stock_by_wh) < 2:
            return action

        next_step_period = Period(self.today, "d", self.days_between_resupplies)
        forecast_by_wh = np.array(
            [self.geu.get_forecast(next_step_period, wh) for wh in self.data_class.wh_list])

        # Calculate balance stock
        stock_by_wh = np.copy(self.stock_by_wh)
        # balance_stock = math.floor(np.sum(stock_by_wh) / stock_by_wh.size)

        remaining_stock = 1
        excess_stock = stock_by_wh - remaining_stock

        # Loop while excess
        while np.amin(stock_by_wh <= remaining_stock) and np.amax(excess_stock) != 0 and np.amin(excess_stock) < 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            # donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= remaining_stock
            excess_stock[max_neg_index] += remaining_stock
            stock_by_wh[max_pos_index] -= remaining_stock
            stock_by_wh[max_neg_index] += remaining_stock

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += 1
            else:
                action[index] -= 1

        remaining_stock = stock_by_wh - forecast_by_wh

        while np.amax(remaining_stock) != 0 and np.amin(remaining_stock) < 0:
            max_neg_index = np.where(remaining_stock == np.amin(remaining_stock))[0][0]
            max_pos_index = np.where(remaining_stock == np.amax(remaining_stock))[0][0]
            amount_donation = min(remaining_stock[max_pos_index], -remaining_stock[max_neg_index])
            remaining_stock[max_pos_index] -= amount_donation
            remaining_stock[max_neg_index] += amount_donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += amount_donation
            else:
                action[index] -= amount_donation
        return action

    def action14(self):
        """Mantener el stock en Bs As. Intentar tener un stock de mes y medio en cada almacén.
        NO hay movimientos hacia Buenos Aires. Redistribuciones Internas"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        next_step_period = Period(self.today, "d", 45)
        forecast_by_wh = np.array(
            [self.geu.get_forecast(next_step_period, wh) for wh in self.data_class.wh_list])

        stock_by_wh = np.copy(self.stock_by_wh)

        donation_bs_as = max(stock_by_wh[0] - math.ceil(forecast_by_wh[0]), 0)
        amount_needed = np.floor(stock_by_wh - forecast_by_wh)

        while np.min(amount_needed) < 0 and donation_bs_as != 0:
            max_neg_index = np.where(amount_needed == np.amin(amount_needed))[0][0]
            donation_bs_as -= 1
            amount_needed[max_neg_index] += 1
            # Add movement to action
            index = self.get_action_index(wh_origin=0, wh_destination=max_neg_index)
            action[index] += 1

        if np.min(amount_needed) < 0:
            while np.amax(amount_needed[1:]) != 0 and np.amin(amount_needed[1:]) < 0:
                amount_needed_without_bs_as = amount_needed[1:]
                max_neg_index = np.where(amount_needed_without_bs_as == np.amin(amount_needed_without_bs_as))[0][0] + 1
                max_pos_index = np.where(amount_needed_without_bs_as == np.amax(amount_needed_without_bs_as))[0][0] + 1
                # amount_donation = min(remaining_stock[max_pos_index], -remaining_stock[max_neg_index])
                amount_donation = 1
                amount_needed[max_pos_index] -= amount_donation
                amount_needed[max_neg_index] += amount_donation

                # Add movement to action
                index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
                if max_pos_index < max_neg_index:
                    action[index] += amount_donation
                else:
                    action[index] -= amount_donation
        return action


    def action14(self):
        """Mantener el stock en Bs As. Intentar tener un stock de mes y medio en cada almacén.
        NO hay movimientos hacia Buenos Aires. Redistribuciones Internas"""
        # Make action space
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        next_step_period = Period(self.today, "d", 45)
        forecast_by_wh = np.array(
            [self.geu.get_forecast(next_step_period, wh) for wh in self.data_class.wh_list])

        stock_by_wh = np.copy(self.stock_by_wh)

        donation_bs_as = max(stock_by_wh[0] - math.ceil(forecast_by_wh[0]) , 0)
        amount_needed = np.floor(stock_by_wh - forecast_by_wh)

        while np.min(amount_needed) < 0 and donation_bs_as != 0:
            max_neg_index = np.where(amount_needed == np.amin(amount_needed))[0][0]
            donation_bs_as -= 1
            amount_needed[max_neg_index] += 1
            # Add movement to action
            index = self.get_action_index(wh_origin=0, wh_destination=max_neg_index)
            action[index] += 1

        if np.min(amount_needed) < 0:
            while np.amax(amount_needed[1:]) != 0 and np.amin(amount_needed[1:]) < 0:
                amount_needed_without_bs_as = amount_needed[1:]
                max_neg_index = np.where(amount_needed_without_bs_as == np.amin(amount_needed_without_bs_as))[0][0] + 1
                max_pos_index = np.where(amount_needed_without_bs_as == np.amax(amount_needed_without_bs_as))[0][0] + 1
                # amount_donation = min(remaining_stock[max_pos_index], -remaining_stock[max_neg_index])
                amount_donation = 1
                amount_needed[max_pos_index] -= amount_donation
                amount_needed[max_neg_index] += amount_donation

                # Add movement to action
                index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
                if max_pos_index < max_neg_index:
                    action[index] += amount_donation
                else:
                    action[index] -= amount_donation
        return action

    def action15(self):
        """
        Mantiene cierto nivel de stock en cada almacen y el resto en 00
        """
        min_stock = 5  # Hardcoded minimum stock
        max_index = int(np.argmax(self.stock_by_wh))
        # get dummy action
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        stock_by_wh = self.stock_by_wh.copy()
        stock_without_main = stock_by_wh.copy()
        stock_without_main[max_index] = min_stock + 100

        available = max(stock_by_wh[max_index] - min_stock, 0)
        amount_needed = max(np.sum(np.clip(min_stock - stock_without_main, 0, 100)), 0)

        if amount_needed <= available:
            for index in range(0, self.num_wh):
                if index != max_index:
                    if index > max_index:
                        action[self.get_action_index(max_index, index)] = max(min_stock - stock_by_wh[index], 0)
                    else:
                        action[self.get_action_index(max_index, index)] = -max(min_stock - stock_by_wh[index], 0)
        else:
            new_min = min_stock
            while amount_needed > available and new_min != 1:
                new_min -= 1
                available = max(stock_by_wh[max_index] - new_min, 0)
                amount_needed = max(np.sum(np.clip(new_min - stock_without_main, 0, 100)), 0)
            if new_min == 1 and amount_needed > available:
                new_available = available
            else:
                new_available = available - amount_needed
                for index in range(0, self.num_wh):
                    if index != max_index:
                        if index > max_index:
                            action[self.get_action_index(max_index, index)] = max(new_min - stock_by_wh[index], 0)
                        else:
                            action[self.get_action_index(max_index, index)] = -max(new_min - stock_by_wh[index], 0)
            if new_available > 0:
                for index in range(0, self.num_wh):
                    if index != max_index:
                        if index > max_index:
                            action[self.get_action_index(max_index, index)] += 1
                            new_available -= 1
                            if new_available == 0:
                                break
                        else:
                            action[self.get_action_index(max_index, index)] -= 1
                            new_available -= 1
                            if new_available == 0:
                                break
        return action

    def action16(self):
        # New action prioritizing wh with old demands
        action = np.array([0] * int(self.num_wh / 2 * (self.num_wh - 1)))

        # If there is no stock to level out then do nothing
        if sum(self.stock_by_wh) == 0:
            return action

        # Calculate balance stock
        stock_by_wh = np.copy(self.stock_by_wh)
        balance_stock = math.floor(np.sum(stock_by_wh) / stock_by_wh.size)
        excess_stock = stock_by_wh - balance_stock
        excedent = np.sum(stock_by_wh) % len(stock_by_wh)
        for i in range(excedent):
            excess_stock[self.data_class.wh_list.index(self.geu.demand_priority[i])] -= 1

        while np.min(excess_stock) < 0:
            # Get maximum and minimum excess values
            max_neg_index = np.where(excess_stock == np.amin(excess_stock))[0][0]
            max_pos_index = np.where(excess_stock == np.amax(excess_stock))[0][0]

            # Donate accordingly
            donation = min(excess_stock[max_pos_index], -excess_stock[max_neg_index])
            excess_stock[max_pos_index] -= donation
            excess_stock[max_neg_index] += donation

            # Add movement to action
            index = self.get_action_index(wh_origin=max_pos_index, wh_destination=max_neg_index)
            if max_pos_index < max_neg_index:
                action[index] += donation
            else:
                action[index] -= donation

    def get_action_index(self, wh_origin: int, wh_destination: int):
        """
        Get action index for the given warehouses
        """
        if wh_origin < self.num_wh and wh_destination < self.num_wh:
            min_index = min(wh_origin, wh_destination)
            max_index = max(wh_origin, wh_destination)
            return int(self.num_wh * min_index - min_index * (min_index + 1) / 2) + max_index - min_index - 1
        else:
            raise KeyError('The combination origin_wh - destination_wh entered is not valid')

    def apply_action(self, action: spaces.Discrete):
        low_level_action = np.array([])

        if action == 0:
            low_level_action = self.action1()
        elif action == 1:
            low_level_action = self.action2()
        elif action == 2:
            low_level_action = self.action3()
        elif action == 3:
            low_level_action = self.action5()
        elif action == 4:
            low_level_action = self.action7()
        elif action == 5:
            low_level_action = self.action13()
        elif action == 6:
            low_level_action = self.action14()
        elif action == 7:
            low_level_action = self.action15()
        elif action == 8:
            low_level_action = self.action16()
        elif action == 9:
            low_level_action = self.action6()
        else:
            print(f"Action no definida {action}")

        low_level_action = np.rint(low_level_action)
        self.sim_redistribute_materials(low_level_action)

        return low_level_action
