import numpy as np
import numpy.linalg as LA
import datetime as dt
import pandas as pd
from typing import Union
import os
from os import listdir
from os.path import isfile, join

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import joblib

from classes.period import Period
from classes.geu import GEU


def list_of_periods(period, first_date, include_period=False):
    """
    Return list of period object, going from newer to older ('descending' order)
    :param period:
    :param first_date:
    :param include_period:
    :return: list_of_periods:
    """
    periods = []
    if include_period:
        current_period = period
    else:
        current_period = period.previous()

    while current_period.last_date >= first_date:
        periods.append(current_period)
        current_period = current_period.previous()

    return periods


def data_for_clustering(geus_by_id: {str: GEU}, date_of_clustering: dt.datetime = dt.datetime(2020, 1, 1),
                        years_clustering: int = 2):
    """
    Cluster GEU objects
    :param geus_by_id: {id, Material}
    :param date_of_clustering: dt.datetime
    :param years_clustering: int
    :return: df for clustering
    """

    # Step 1: Data preparation

    # List of columns of the df
    geu_id = []
    weighted_demand_v1_tkt = []

    # Make period lists
    period = Period(date_of_clustering, 'y', 1)
    min_date = period.first_date - dt.timedelta(days=years_clustering * 365)
    periods = list_of_periods(period, min_date)

    # Step 2: Calculate features

    # Iterate through all GEUs
    for geu in geus_by_id.values():
        # GEU id
        geu_id.append(geu.id)

        # Tickets

        # Leave tickets that are within specified range
        tickets_in_range = {}
        for date, tickets in geu.tickets.items():
            if min_date <= date < date_of_clustering:
                tickets_in_range[date] = tickets

        # ------------------

        # Tickets
        wd_v1 = 0

        # geu.tickets[date] is a list of tickets, no matter the len
        # geu.tickets[date][0] is a ticket from a certain GREP
        # geu.tickets[date][0].amount is the ticket amount: 1
        # geu.tickets[date][0].grep is the ticket grep: '01'

        # Go through all tickets
        for date, tickets in geu.tickets.items():
            # Enter a certain period
            for i, period in enumerate(periods):
                # If there's a match, go to next ticket
                if period.contains(date):
                    # Go through all tickets from that date
                    for tkt in tickets:
                        # v1: No weights
                        wd_v1 += tkt.amount
                    break

        # Append aggregated values
        weighted_demand_v1_tkt.append(wd_v1)

    # ----------------------

    # Step 3: Save features information

    # Make df with data
    columns_name = ['GEU', 'Weighted Demand v1 (TKT)']
    data = list(zip(geu_id, weighted_demand_v1_tkt))
    df = pd.DataFrame(data, columns=columns_name)

    # Save features in attributes
    for index, row in df.iterrows():
        geus_by_id[row['GEU']].weighted_demand_v1_tkt = row['Weighted Demand v1 (TKT)']

    return df


def cluster_geus(df: pd.DataFrame):
    """
    Cluster GEU objects
    :param df: pd.DataFrame
    :return: clustered GEUs
    """

    # Cluster assignment
    def assign_cluster(x):
        if x == 0:      # No demand in 2 years
            return 0
        elif x <= 3:    # Low demand
            return 1
        elif x <= 24:   # Medium demand (1 per month)
            return 2
        else:           # High demand
            return 3

    # Calculate low, medium and high demand
    df['label'] = df['Weighted Demand v1 (TKT)'].apply(assign_cluster)

    # Set fixed number of clusters
    clusters = 4

    return df, clusters
