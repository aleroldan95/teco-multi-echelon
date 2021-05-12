import numpy as np
import pandas as pd
import datetime as dt
import os
import time
import warnings
import pickle as pkl
warnings.simplefilter("ignore")

import streamlit as st
from sqlalchemy import create_engine, exc
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import SimpleExpSmoothing

# Data Base Connections-------------------------------------------------------------------------------------------------


def postgres_connection(is_streamlit: bool = False, print_on: bool = True):
    """
    Create or close connection to PostgreSQL server
    :return:
    """
    tic = elapsed_time_message(True)                                                         # Start Timer
    # --------------------------------------------------------------------------------------------------------------

    # create databases
    databases = {
        'production': {
            'NAME': 'TelecomMultiechelon',
            'USER': 'postgres',
            'PASSWORD': 'continente7',
            'HOST': '34.233.129.172',
            'PORT': 18081,
        },
        'teco': {
            'NAME': 'TelecomMultiechelon',
            'USER': 'postgres',
            'PASSWORD': 'NEURALTeco',
            'HOST': '127.0.0.1',
            'PORT': 5432,
        }
    }
    # choose the database to use
    db = databases['production']
    # construct an engine connection string
    engine_string = "postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}".format(
        user=db['USER'],
        password=db['PASSWORD'],
        host=db['HOST'],
        port=db['PORT'],
        database=db['NAME'],
    )
    # create sqlalchemy engine
    conn = create_engine(engine_string)

    # --------------------------------------------------------------------------------------------------------------
    message = "Creating Data Base connection"
    elapsed_time_message(False, message, tic, print_on=print_on, is_streamlit=is_streamlit)

    c = conn.connect()
    try:
        # suppose the database has been restarted.
        c.execute("SELECT * FROM produccion.materiales LIMIT 1")
        c.close()
    except exc.DBAPIError:
        print("Connection was invalidated!")
    except conn:
        # an exception is raised, Connection is invalidated.
        if conn.connection_invalidated:
            print("Connection was invalidated!")

    return conn


def create_query(query, engine):
    """
    Create a query for the SQL Server
    :param query: Insert the query between comillas
    :param engine: sql connection
    :return: Dataframe with the query result
    """
    return pd.io.sql.read_sql(query, con=engine)


def run_query(query, engine):
    try:
        with engine.connect() as conn:
            conn.execute(query)
    except Exception as e:
        raise Exception(e)


def get_pickle(filename):
    path = os.path.join(os.path.dirname(__file__), 'pickles/' + filename)
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj


def upload_from_dataframe(df, engine, table_name, schema='public', if_exists='replace', chunksize=1000000):
    try:
        df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False, chunksize=chunksize)
    except Exception as e:
        raise Exception(e)


# Print Messages--------------------------------------------------------------------------------------------------------

def elapsed_time_message(is_start: bool, message: str = None, tic=None, is_end: bool = False,
                         is_streamlit: bool = False, print_on: bool = True):
    """
    Prints elapsed time of a process as a formatted message
    :param is_start: Select start or finish message
    :param message: print message
    :param tic: time.time()
    :param is_end: Select end message
    :param is_streamlit: bool
    :param print_on: bool
    :return:
    """
    if is_end:
        toc = time.time()
        if print_on:
            print(f"\n{message}. Total Run Time: {round(toc - tic, 4)} seconds.")
        if is_streamlit:
            st.write(f"\n{message}. Total Run Time: {round(toc - tic, 4)} seconds.")

    elif is_start:
        tic = time.time()
        return tic
    else:
        toc = time.time()
        time_delta = toc - tic

        secs = str(round(time_delta, 4))
        len_secs = len(secs)
        secs += (5 - (len_secs - secs.find('.'))) * "0"
        n_spaces = 39 - len(message)
        len_secs = len(secs)

        printable = message + " " * n_spaces + "[Done] Process time:" + (7 - len_secs) * " " + secs + " seconds"
        if is_streamlit:
            st.write(printable)
        if print_on:
            print(printable)


def main_timer(start: bool, tic=0, is_streamlit=False):
    if start:
        # Start main timer
        tic = time.time()
        return tic
    else:
        toc1 = time.time()
        if is_streamlit:
            st.success(f"Algorithm Finished. Total Run Time: {round(toc1 - tic, 4)} seconds.")
        else:
            print(f"Algorithm Finished. Total Run Time: {round(toc1 - tic, 4)} seconds.")


def to_datetime(date):
    """
    Converts to datetime dtype
    :param date:
    :return:
    """

    if isinstance(date, pd.Timestamp) or isinstance(date, dt.datetime):
        return dt.datetime(date.year, date.month, date.day)
    elif isinstance(date, str):
        if "/" in date:
            split_date = date.split("/")
            return dt.datetime(int(split_date[2]), int(split_date[1]), int(split_date[0]))
        else:
            split_date = date.split("-")
            day = split_date[2].split(" ")
            return dt.datetime(int(split_date[0]), int(split_date[1]), int(day[0]))


def linear_forecast(mat_object, period, first_date, warehouse, soft=False):
    """
    Linear forecast for material demand. 'soft' parameter is used for deleting outliers.
    :param mat_object:
    :param period:
    :param first_date:
    :param warehouse:
    :param soft:
    :return:
    """

    periods = list_of_periods(period, first_date)
    dates = [p.first_date for p in periods]
    values = [mat_object.get_demand(p, warehouse) for p in periods]

    dates.reverse()
    values.reverse()

    if len(values) == 2:
        return sum(values)/2

    if soft:
        non_outlier_list = deleting_outliers(values)
        filtered_dates = []
        filtered_values = []
        for i in range(len(values)):
            if values[i] in non_outlier_list:
                filtered_dates.append(dates[i])
                filtered_values.append(values[i])
        x_df = pd.to_datetime(filtered_dates)
        y = np.array(filtered_values)
    else:
        x_df = pd.to_datetime(dates)
        y = np.array(values)
    x_df = x_df.map(dt.datetime.toordinal)
    x = np.array(x_df).reshape((-1, 1))

    model = LinearRegression()
    try:
        model.fit(x, y)
    except:
        print(x, mat_object, period, first_date, warehouse)

    return model.predict([[dt.datetime.toordinal(period.first_date)]])

def expo_forecast(mat_object, period, first_date, warehouse, soft=False):
    """
    Linear forecast for material demand. 'soft' parameter is used for deleting outliers.
    :param mat_object:
    :param period:
    :param first_date:
    :param warehouse:
    :param soft:
    :return:
    """
    ALPHA = 0.7

    periods = list_of_periods(period, first_date)
    dates = [p.first_date for p in periods]
    values = [mat_object.get_demand(p, warehouse) for p in periods]

    dates.reverse()
    values.reverse()

    if len(values) == 1:
        return values[0]

    if soft:
        non_outlier_list = deleting_outliers(values)
        filtered_dates = []
        filtered_values = []
        for i in range(len(values)):
            if values[i] in non_outlier_list:
                filtered_dates.append(dates[i])
                filtered_values.append(values[i])
        x_df = pd.to_datetime(filtered_dates)
        y = np.array(filtered_values)
    else:
        x_df = pd.to_datetime(dates)
        y = np.array(values)
    x_df = x_df.map(dt.datetime.toordinal)
    x = np.array(x_df).reshape((-1, 1))

    try:
        model = SimpleExpSmoothing(y).fit(optimized=False, smoothing_level=ALPHA)
    except:
        print(x, mat_object, period, first_date, warehouse)
        return

    return model.forecast(1)[0]


def list_of_periods(period, first_date, include_period=False):
    """
    Return list of period object
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


def deleting_outliers(list_of_amounts):
    """
    Deletes outliers from a time series. It can be used when forecasting demand.
    :param list_of_amounts:
    :return:
    """

    if len(list_of_amounts) == 0:
        return list_of_amounts
    quartile_1, quartile_3 = np.percentile(list_of_amounts, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    list_of_amounts = [amount for amount in list_of_amounts if amount >= lower_bound]
    list_of_amounts = [amount for amount in list_of_amounts if amount <= upper_bound]
    return list_of_amounts


def process_initial_stock(materials_by_id, geus_by_id, string_date, warehouse_list, is_cram=False):
    # Start Timer
    tic = time.time()
    message = "Set starting stock..."
    # ------------------------------------------------------------------------------------------------------------------

    engine = postgres_connection()
    # Query: initial stock (matching the string_date)
    if is_cram:
        query = f"""select material, centro, fecha, stock from p03_stock_by_date_grep where fecha = '{string_date}'"""
    else:
        query = f"""select material, grep, fecha, stock from p03_stock_by_date_grep where fecha = '{string_date}'"""
    starting_stock_df = create_query(query, engine)

    # Set default initial stock as 0
    for geu in geus_by_id.values():
        geu.set_zero_stock(warehouse_list)

    # Set starting stock for the given initial date
    for index, row in starting_stock_df.iterrows():
        if row['material'] in materials_by_id.keys():
            materials_by_id[row['material']].geu.set_starting_stock_from_row(row, is_teco_simulation=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Stop Timer
    toc = time.time()
    elapsed_time_message(toc - tic, message)




def process_compras(geus_by_id, filename = '7crams_1_compra_2020-01-02_low'):

    df_compras = pd.read_csv(
        f'C:/Users/alejandro.d.roldan/Accenture/Telecom Inventory Optimization - Multi-Echelon - 07. Resultados/Corridas2/{filename}.csv',
    dtype={'GEU':str})
    df_compras['heuristica'] = 'NeuralNet'
    df_compras['fecha'] = dt.datetime(2020,10,29)
    df_compras.rename(columns={'GEU': 'geu',
                               'Cantidad Requerida':'compras'}, inplace=True)

    df_entradas = pd.io.sql.read_sql("SELECT * from p02_entradas", con=postgres_connection(print_on=False))
    df_entradas['pcompras'] = df_entradas["compras"] / (
            df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])
    df_entradas['preparaciones'] = df_entradas["reparaciones"] / (
            df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])
    df_entradas['pdesmontes'] = df_entradas["desmontes"] / (
            df_entradas["compras"] + df_entradas['reparaciones'] + df_entradas["desmontes"])

    df_new_compras = pd.DataFrame(
        columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm", "ingresos_manuales"])
    for _, row in df_compras.iterrows():
        geu_id = row["geu"]
        if geu_id not in df_entradas["geu"].values:
            if geus_by_id[geu_id].procurement_type == "SPM":
                df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, 0, 0, row["compras"], 0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                          "spm", "ingresos_manuales"]))
            elif geus_by_id[geu_id].procurement_type == "Buyable":
                df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"], 0, 0, 0, 0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                          "spm", "ingresos_manuales"]))
            elif geus_by_id[geu_id].procurement_type == "Repairable":
                df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, row["compras"], 0, 0, 0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                          "spm", "ingresos_manuales"]))
            else:
                df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], 0, 0, row["compras"], 0, 0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes",
                                          "spm", "ingresos_manuales"]))
        else:
            if geus_by_id[geu_id].procurement_type == "SPM":
                df_new_compras = df_new_compras.append(
                    pd.DataFrame([(row["geu"], row["heuristica"], row["fecha"], row["compras"] *
                                   df_entradas.loc[df_entradas['geu'] == row["geu"], 'pcompras'].iloc[0],
                                   row["compras"] *
                                   df_entradas.loc[df_entradas['geu'] == row["geu"], 'preparaciones'].iloc[0], 0,
                                   row["compras"] *
                                   df_entradas.loc[df_entradas['geu'] == row["geu"], 'pdesmontes'].iloc[0],
                                   0)],
                                 columns=["geu", "heuristica", "fecha", "compras", "reparaciones", "desmontes", "spm",
                                          "ingresos_manuales"]), ignore_index=True)
            elif geus_by_id[geu_id].is_dismountable:
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
    df_new_compras.to_csv(
        f'C:/Users/alejandro.d.roldan/Accenture/Telecom Inventory Optimization - Multi-Echelon - 07. Resultados/Corridas2/new_compras_{filename}.csv')
    return df_new_compras