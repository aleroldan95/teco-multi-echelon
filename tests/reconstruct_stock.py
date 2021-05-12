import time
from functions import elapsed_time_message, postgres_connection, create_query, main_timer
import datetime as dt
import pandas as pd
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------


def generate_stock_per_date(materials_by_id, first_date, last_date, table_sql_movements='r00', movements=[]):
    tic = main_timer(True)

    print("\n")
    conn = postgres_connection()

    # Load movements and add them to each material as attributes
    df_movements = load_movements(materials_by_id, conn, table_sql_movements, movements)
    adding_movements_to_stock(materials_by_id, df_movements, first_date, last_date)

    # Load initial stock and add them to each material
    df_zero_stock = load_zero_stock(materials_by_id, conn)
    generate_stock_zero(df_zero_stock, materials_by_id)

    # Calculate stock
    calculate_stock_per_date(materials_by_id, first_date, last_date)
    df_stock = generate_total_stock(materials_by_id, first_date, last_date)

    main_timer(False, tic)

    return df_zero_stock, df_movements, df_stock


# ----------------------------------------------------------------------------------------------------------------------

def load_zero_stock(materials_by_id, conn):
    tic = time.time()
    message = "Loading zero stock..."

    # Query: load stock
    query = """SELECT * FROM p01_stock_02_01_2020"""
    df_zero_stock = create_query(query, conn)

    # Must be in scope
    list_of_materials_in_scope = list(materials_by_id.keys())
    df_zero_stock = df_zero_stock[df_zero_stock["material"].isin(list_of_materials_in_scope)]

    # Almacen cant be empty
    df_zero_stock = df_zero_stock[df_zero_stock["almacen"] != ""]
    # Rename almacen with its centro
    df_zero_stock["almacen"] = df_zero_stock["centro"] + '-' + df_zero_stock["almacen"]

    toc = time.time()
    elapsed_time_message(toc - tic, message)

    return df_zero_stock


def generate_stock_zero(df_zero_stock, materials_by_id, zero_stock_date=dt.datetime(2020, 1, 2)):
    tic = time.time()
    message = "Assigning zero stock..."

    # Aggregate df stock by material & almacen
    group_by_mat_alm = df_zero_stock.groupby(["material", "almacen"])["cantidad"].sum().reset_index()

    # Adds corresponding stock as an attribute of the material
    [materials_by_id[material].set_stock(zero_stock_date, warehouse, amount)
     for material, warehouse, amount in group_by_mat_alm[["material", "almacen", "cantidad"]].values]

    # Set value for each material & warehouse combination
    for material in materials_by_id:
        for warehouse in materials_by_id[material].warehouses:
            try:
                materials_by_id[material].stock[(zero_stock_date, warehouse)]
            except:
                materials_by_id[material].set_stock(zero_stock_date, warehouse, 0)

    toc = time.time()
    elapsed_time_message(toc - tic, message)


def load_movements(materials_by_id, conn, table_sql_movements, movements):
    tic = time.time()
    message = "Leyendo movimientos..."

    # Processed table
    if table_sql_movements == 'p02':
        df_movements = create_query("""SELECT * FROM p02_total_historic_movements""", conn)

        # Materials must be in scope
        df_movements = df_movements[df_movements["material"].isin(list(materials_by_id.keys()))]
        # If applies, use movements list to filter movements in df_movements
        if movements:
            df_movements = df_movements[df_movements["clase_de_movimiento"].isin(movements)]

        # Convert date column to datetime
        df_movements["fecha_de_documento"] = pd.to_datetime(df_movements["fecha_de_documento"])

        # Reduce movements to bare minimum
        df_movements = df_movements.groupby(["material", "fecha_de_documento", "almacen"])[
            "cantidad"].sum().reset_index()

    # Raw table
    if table_sql_movements == 'r00':
        # Query
        df_movements = create_query("""SELECT material, centro, almacen, fe_contabilizacion, 
        ctd_en_um_entrada, clase_de_movimiento FROM r00_total_historic_movements""", conn)

        df_movements.columns = ["material", "centro", "almacen", "fecha_de_documento", "cantidad",
                                "clase_de_movimiento"]

        # Materials must be in scope
        df_movements = df_movements[df_movements["material"].isin(list(materials_by_id.keys()))]

        # Almacen cant be empty
        df_movements = df_movements[df_movements["almacen"] != ""]
        # Rename almacen with its centro
        df_movements["almacen"] = df_movements["centro"] + '-' + df_movements["almacen"]

        # Convert date column to datetime and quantity column to float
        df_movements["fecha_de_documento"] = pd.to_datetime(df_movements["fecha_de_documento"])
        df_movements["cantidad"] = df_movements["cantidad"].astype('float64')

        # If applies, use movements list to filter movements in df_movements
        if movements:
            df_movements = df_movements[df_movements["clase_de_movimiento"].isin(movements)]

        # Reduce movements to bare minimum
        df_movements = df_movements.groupby(["material", "fecha_de_documento", "almacen"])[
            "cantidad"].sum().reset_index()

    toc = time.time()
    elapsed_time_message(toc - tic, message)

    return df_movements


def adding_movements_to_stock(materials_by_id, df_movements, first_date, last_date):
    tic = time.time()
    message = "Adding movements..."

    # If the date is between the corresponding date limits,
    # the movement and the corresponding warehouses are added to the material attributes
    [materials_by_id[material].set_movement(date, warehouse, amount)
     for material, warehouse, date, amount
     in df_movements[["material", "almacen", "fecha_de_documento", "cantidad"]].values
     if first_date <= date <= last_date]

    toc = time.time()
    elapsed_time_message(toc - tic, message)


def calculate_stock_per_date(materials_by_id, first_date, last_date):
    tic = time.time()
    message = "Generando stock por día..."

    # Hardcoded stock date reference
    date = dt.datetime(2020, 1, 2)

    # Iterate through dates and correctly set material stock according to previous stock and new movements
    if last_date >= date:
        while date != last_date:
            for material in materials_by_id.values():
                for warehouse in material.warehouses:
                    try:
                        amount = material.movements[(date, warehouse)]
                    except:
                        amount = 0
                    date_stock = material.get_stock(date, warehouse) + amount
                    material.set_stock(date + dt.timedelta(days=1), warehouse, date_stock)

            date = date + dt.timedelta(days=1)

    # Hardcoded stock date reference
    date = dt.datetime(2020, 1, 2)

    # Iterate through dates and correctly set material stock according to previous stock and new movements
    if first_date <= date:
        while date != first_date:
            for material in materials_by_id.values():
                for warehouse in material.warehouses:
                    try:
                        amount = material.movements[(date, warehouse)]
                    except:
                        amount = 0
                    date_stock = material.get_stock(date, warehouse) - amount
                    material.set_stock(date - dt.timedelta(days=1), warehouse, date_stock)

            date = date - dt.timedelta(days=1)

    toc = time.time()
    elapsed_time_message(toc - tic, message)


def generate_total_stock(materials_by_id, first_date, last_date):
    tic = time.time()
    message = "Generating exportable dataframe..."

    date = []
    warehouse = []
    amount = []
    mat = []

    # Add all info to the respective lists
    for material in materials_by_id.values():
        for k, v in material.stock.items():
            if first_date <= k[0] <= last_date:
                date.append(k[0])
                warehouse.append(k[1])
                amount.append(v)
                mat.append(material.catalog)

    # Convert to DataFrame
    df = pd.DataFrame({'Material': mat, 'Fecha': date, 'Almacén': warehouse, 'Cantidad': amount})

    toc = time.time()
    elapsed_time_message(toc - tic, message)

    return df
