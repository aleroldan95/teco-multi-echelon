import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)

import pandas as pd
import os
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pickle as pkl
import numpy as np
import ray
import base64
import matplotlib.pyplot as plt
from datetime import timedelta

from functions import postgres_connection,  create_query
from classes.stock_in_transit import StockInTransit
from reinforcement_learning.order_env import OrderEnvironment
import ray.rllib.agents.impala as impala
from ray.tune import function
from data_loader_class import DataClass
import reinforcement_learning.optimizador_de_compras as opt
from reinforcement_learning.redistribution_env import SimulationEnvironment


#-----------------------------------------------------------------------------------------------------------------------

@st.cache(allow_output_mutation=True)
def connection_to_server(is_streamlit):
    return postgres_connection(is_streamlit=False)


def load_sql_table(name):
    # Query material table in db
    query = f"""SELECT * FROM {name};"""
    df = create_query(query, postgres_connection(is_streamlit=False))
    return df

@st.cache
def load_grep_geometry(grep_type):
    # Create query
    query = f"""select grep, lat, lgt from r00_relation_grep_warehouse where original='1';"""
    df_grep = create_query(query, postgres_connection(is_streamlit=False))
    df_grep = df_grep.groupby(['grep']).first().reset_index()
    df_grep.columns = ["grep", 'lat', 'long']
    if grep_type=='7_crams':
        query="""select grep, right(centro,2) as cram from p01_grep_centro where centro like 'C%'"""
        relation_grep_cram = create_query(query, postgres_connection(is_streamlit=False))
        df_grep = pd.merge(df_grep,relation_grep_cram,left_on='grep',right_on='grep', how='inner')
        df_grep.index = df_grep.cram
        df_grep.drop(['grep', 'cram'], axis=1, inplace=True)
        return df_grep
    df_grep.index = df_grep.grep
    df_grep.drop(['grep'], axis=1, inplace=True)
    return df_grep

def end_page():
    st.write('---')
    st.markdown("<h6 style='text-align: center; '>*Supply Brain by Accenture (v1.0.0)*</h6>", unsafe_allow_html=True)

def load_last_date_stock():
    query = """select max(fecha) from produccion.stock_s4"""
    return create_query(query,  postgres_connection(is_streamlit=False))

def load_stock_dates():
    query = """select distinct fecha from produccion.stock_s4"""
    df = create_query(query, postgres_connection(is_streamlit=False))
    return sorted(list(df.fecha.unique()))

def set_multiplier_to_deafult(type, resupplies_per_year, grep_type):
    with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{grep_type}_{resupplies_per_year}_{type}.pkl'), "rb") as file:
        multiplicadores = pkl.load(file)

    #multiplicadores = {'ACCESO - FIJA': [(1, 3), (1, 6), (1.2, 2)],
    #                   'ACCESO - MOVIL': [(1, 2), (1, 1), (1, 0)],
    #                   'CORE VOZ': [(1, 2), (1, 1), (1, 0)],
    #                   'ENTORNO': [(1, 2), (1, 4), (1.1, 1)],
    #                   'TRANSPORTE - DX': [(1, 2), (1, 3), (1, 1)],
    #                   'TRANSPORTE - TX': [(1, 2), (1, 3), (1, 0)],
    #                   'TX - RADIOENLACES Y SATELITAL': [(1, 3), (1, 2), (1.1, 1)],
    #                   'CMTS': [(1, 0), (1, 0), (1, 0)]}

    opt.save_multipliers(multiplicadores=multiplicadores, is_streamlit=True, resupplies_per_year=None, grep_type=None)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}"><input type="button" value="Descargar {file_label}"></a>'
    return href


def show_update_options(name, sql_name, types):
    st.header(name)
    if st.checkbox(f'Mostrar {name}'):
        df_query = load_sql_table(sql_name)
        st.dataframe(df_query.tail())
        show_dowload_option(df_query, name)

    new_data = st.file_uploader(f"Cargar nuevos {name}", type=("xlsx"))
    update_new_data(df_picked=new_data, name=sql_name, columns_types=types)
    st.markdown('----')

def set_neuronal_config(env_config):
    ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
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
    trainer = impala.impala.ImpalaTrainer(config)
    trainer.restore(
        "../reinforcement_learning/models/IMPALA/IMPALA_SimulationEnvironment_0_2020-10-08_20-55-26efyv3ax6/checkpoint_2200/checkpoint-2200")

    return trainer

def close_neuronal_connection():
    ray.shutdown()

def show_dowload_option(df, name, is_sidebar=False):
    style = """text-decoration:none; 
                        background-color:#ffffff; 
                        padding-top: 10px; padding-bottom: 10px; padding-left: 25px; padding-right: 25px;
                        box-shadow: 4px 4px 14px rgba(0, 0, 0, 0.25); 
                        font-family: Montserrat;
                        font-style: normal;
                        font-weight: 600;
                        font-size: 14px;
                        line-height: 13px;
                        text-align: center;
                        color: #FF037C;"""
    STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
    DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
    if not DOWNLOADS_PATH.is_dir():
        DOWNLOADS_PATH.mkdir()

    if os.path.exists(str(DOWNLOADS_PATH / f"{name}.xlsx")):
        os.remove(str(DOWNLOADS_PATH / f"{name}.xlsx"))

    df.to_excel(str(DOWNLOADS_PATH / f"{name}.xlsx"), index=False)

    if is_sidebar:
        st.sidebar.markdown(
            f'<a href="downloads/{name}.xlsx" download="{name}.xlsx" style="{style}">Descargar Archivo Excel</a>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<a href="downloads/{name}.xlsx" download="{name}.xlsx" style="{style}">Descargar Archivo Excel</a>',
            unsafe_allow_html=True)

def generate_exportable_redistribution(distributions:{str:StockInTransit}, state):
    geus = []
    donator = []
    receiving = []
    amount = []
    lt = []
    crti = []
    domain = []
    name = []
    type = []
    catalogs = []
    cluster = []
    marca = []
    equipo = []

    for geu_id, movements in distributions.items():
        for mov in movements:
            catalogs.append([material.catalog for material in state.geus_by_id[geu_id].materials])
            geus.append(geu_id)
            donator.append(mov.donating_wh)
            receiving.append(mov.receiving_wh)
            amount.append(int(mov.amount))
            lt.append(mov.leadtime)
            crti.append(state.geus_by_id[geu_id].criticality)
            domain.append(state.geus_by_id[geu_id].domain)
            name.append(state.geus_by_id[geu_id].name)
            type.append(state.geus_by_id[geu_id].procurement_type)
            cluster.append(state.geus_by_id[geu_id].cluster)
            marca.append(state.geus_by_id[geu_id].brand)
            equipo.append(state.geus_by_id[geu_id].equipment)

    result = pd.DataFrame({
        'GEU': geus,
        'Descripción': name,
        'Catálogos':catalogs,
        'Origen': donator,
        'Destino': receiving,
        'Cantidad': amount,
        'Leadtime Estimado': lt,
        'Criticidad': crti,
        'Dominio': domain,
        'Marca':marca,
        'Equipo':equipo,
        'Tipo de Abastecimiento': type,
        'Cluster':cluster
    })
    result.Cluster = result.Cluster.apply(
        lambda x: str(x) + ' (baja)' if x == 1 else str(x) + ' (media)' if x == 2 else str(x) + ' (alta)')

    return result


def get_geu_stock(state, geu):
    state.env.geu = geu
    state.env.reset()
    return sum(state.env.stock_by_wh)

def generate_mat_geu_dictionary(state):
    return pd.DataFrame(
        [(mat.catalog,
          mat.name,
          mat.geu.id) for mat in state.DataClass.materials_by_id.values()],
        columns=['Material',
                 'Descripcion',
                 'GEU'
                 ])

def generate_geu_dictionary(state):
    return pd.DataFrame(
                    [(geu.id,
                      geu.name,
                      geu.brand,
                      geu.area,
                      [material.catalog for material in geu.materials],
                      geu.domain,
                      geu.criticality,
                      geu.price,
                      geu.price_unit,
                      geu.rep_price,
                      geu.rep_unit,
                      str(geu.leadtime) + ' +/- ' + str(geu.leadtime_sd),
                      geu.cluster,
                      geu.is_spm,
                      geu.is_buyable,
                      geu.is_repairable,
                      geu.is_dismountable,
                      geu.procurement_type,
                      #get_geu_stock(state,geu)
                      ) for geu in state.geus_by_id.values()],
                             columns=['GEU',
                                      'Descripcion',
                                      'Marca',
                                      'Area',
                                      'Materiales',
                                      'Dominio',
                                      'Criticidad',
                                      'Precio',
                                      'Unidad Precio',
                                      'Precio Reparaciones',
                                      'Unidad Reparaciones',
                                      'Leadtime',
                                      'Cluster',
                                      'SPM',
                                      'Comprable',
                                      'Reparable',
                                      'Desmontable',
                                      'Tipo_de_Abastecimiento',
                                      #'Stock'
                                      ])

def policy_mapping_fn(agent_id):
    if agent_id.startswith("low_level_"):
        return "low_level_policy"
    else:
        return "high_level_policy"

def initialize_neuronal_environment():
        env = OrderEnvironment()

        config = impala.DEFAULT_CONFIG.copy()
        config["env"] = OrderEnvironment
        config["num_workers"] = 0
        config["rollout_fragment_length"] = 52
        config["log_level"] = "WARNING"
        config["framework"] = "tf"
        config["multiagent"] = {
            "policies": {
                "high_level_policy": (None, env.observation_space,
                                      env.action_space, {
                                          "gamma": 0.9
                                      }),
                "low_level_policy": (None,
                                     env.distribution_env.observation_space,
                                     env.distribution_env.action_space, {
                                         "gamma": 0.9
                                     }),
            },
            "policy_mapping_fn": function(policy_mapping_fn),
        }
        trainer = impala.ImpalaTrainer(config, OrderEnvironment)
        trainer.restore(
            "../reinforcement_learning/models/IMPALA/IMPALA_OrderEnvironment_0_2020-09-14_16-00-246k07r70b/checkpoint_40/checkpoint-40")

        return env, trainer

def translate_action(action, env):

    # tuple (i, j)
    # i: donating warehouse
    # j: receiving warehouse
    # index_count : action numpy index
    # action is a numpy array

    index_count = 0

    movements = []

    for i in range(len(env.data_class.wh_list) - 1):
        for j in range(i + 1, len(env.data_class.wh_list)):
            value = action[index_count]

            if value > 0 and env.stock_by_wh[i] > 0:
                donation = min(env.stock_by_wh[i], value)
                if donation > 0:
                    env.stock_by_wh[i] -= donation
                    leadtime = env.data_class.leadtime_between_warehouses.get(
                        (env.data_class.wh_dictionary[i], env.data_class.wh_dictionary[j]), 4)

                    movements.append(StockInTransit(receiving_wh=env.data_class.wh_dictionary[j],
                                                     date=env.today,
                                                     amount=donation,
                                                     leadtime=leadtime,
                                                     donating_wh=env.data_class.wh_dictionary[i]))

            elif value < 0 and env.stock_by_wh[j] > 0:
                donation = min(env.stock_by_wh[j], - value)
                if donation > 0:
                    env.stock_by_wh[j] -= donation
                    leadtime = env.data_class.leadtime_between_warehouses.get(
                        (env.data_class.wh_dictionary[j], env.data_class.wh_dictionary[i]), 4)

                    movements.append(StockInTransit(receiving_wh=env.data_class.wh_dictionary[i],
                                                     date=env.today,
                                                     amount=donation,
                                                     leadtime=leadtime,
                                                     donating_wh=env.data_class.wh_dictionary[j]))
            index_count += 1

    return movements

def update_new_data(df_picked, name:str, columns_types: [str]):
    """
    Convert .csv or .xlsx in pandas DataFrame. Show upload options and upload data
    :param df_pick: datafram user input
    :param name: sql table' name to insert new data
    :param columns_types: list of columns types
    :return:
    """
    if df_picked:
        try:
            df_picked.seek(0)
            if name == 'produccion.materiales':
                df = pd.read_excel(df_picked, dtype={'Catalogo_S4': 'str', 'Catalogo':'str'})
            elif name == 'produccion.equivalencias':
                df = pd.read_excel(df_picked, dtype={'CatS4': 'str', 'EquivS4': 'str'})
                df['CatS4'] = df['CatS4'].replace('.0','')
                df['EquivS4'] = df['EquivS4'].replace('.0', '')
            elif name == 'produccion.stock_s4':
                df = pd.read_excel(df_picked, dtype={'material': 'str'})
                df['material'] = df['material'].replace('.0', '')
            elif name == 'produccion.tickets_s4':
                df = pd.read_excel(df_picked, dtype={'material': 'str'})
                df['material'] = df['material'].replace('.0', '')
            else:
                df = pd.read_excel(df_picked)
            df.fillna('null', inplace=True)
            # st.success('Carga Finalizada')
            st.dataframe(df.head(3))
            st.markdown('### :raised_hand:Seleccione una opción de Actualización')
            update_option = st.radio(label='',
                                     options=['Cargar Nuevos Datos', 'Borrar y Recargar Datos'],
                                     index=0,
                                     key=name)

            update_data = st.button("Actualizar", key=name)
            if update_data:
                is_valid = True
                if name == 'produccion.greps_habilitados':
                    is_valid, valid_greps = check_greps_names(list(df.grep))
                    if not is_valid:
                        st.error(f'La lista de greps ingresada no coincide con los greps originales. Corroborar nombres, '
                                 f'los greps válidos son los siguientes: {valid_greps}. Sinó cambiar los greps en Relación GREP-Almacén')
                if is_valid:
                    try:
                        conn = connection_to_server(is_streamlit=False)
                        if update_option == 'Cargar Nuevos Datos':
                            insert_new_data(df, name, columns_types, conn)
                        elif update_option == 'Borrar y Recargar Datos':
                            conn.execute(f"""DELETE FROM {name}""")
                            insert_new_data(df, name, columns_types, conn)
                    except:
                        st.error(
                            'No se pudo actualizar. Revisar cantidad de columnas o tipo de datos de la tabla ingresada.')

        except:
            st.warning('No se pudo cargar correctamenta la data. Compruebe que sea un excel')

def check_greps_names(greps_list):
    df = load_sql_table("produccion.relacion_grep_almacen")
    return sum([True for gr in greps_list if gr in list(df.grep.unique())]) == len(greps_list), list(df.grep.unique())

def insert_new_data(df, name, columns_types, conn):

    aux = []
    for row in df.values:
        aux.append(row)

    for i in range(len(aux)):
        row = aux[i]
        aux[i] = '('
        for x, word in enumerate(row):
            if x == len(row) - 1:
                if columns_types[x] in ['String', 'Date']:
                    aux[i] += '\'' + str(word) + '\')'
                else:
                    aux[i] += str(word) + ')'
            else:
                if columns_types[x] in ['String', 'Date']:
                    aux[i] += '\'' + str(word) + '\','
                else:
                    aux[i] += str(word) + ','

    final = ''
    for i in aux:
        final += i + ','
    final = final[:-1]

    query = f"""INSERT INTO {name} VALUES {final} """

    conn.execute(query)

    st.success('Actualización Finalizada')

def generate_pbi_geus_info(geus):
    # GEUS
    keys_list = ['geu', 'cluster']
    keys_list_dict = ['geu', 'cluster', 'descripcion', 'dominio', 'subarea', 'marca', 'equipo',
                      'completo', 'precio', 'unidad precio', 'precio reparacion', 'unidad reparacion',
                      'tipo de precio', 'leadtime', 'criticidad', 'tipo de abastecimiento']
    df_dict = pd.DataFrame(columns=keys_list_dict)

    # Iterate through all GEUs
    for geu in geus.values():
        # Fetch values
        values_list = [geu.id, geu.cluster]
        # Make df with GEU info

        # Fetch values
        values_list_dict = [geu.id, geu.cluster, geu.name, geu.domain, geu.subarea, geu.brand, geu.equipment,
                            geu.is_strongly_connected, geu.price, geu.price_unit, geu.rep_price, geu.rep_unit,
                            geu.price_type, geu.leadtime, geu.criticality, geu.procurement_type]
        # Make df with GEU info
        df_dict = df_dict.append(dict(zip(keys_list_dict, values_list_dict)), ignore_index=True)

    #df_dict.to_csv(os.path.join(os.path.dirname(__file__), relative_path + 'pbi_geus_info.csv'), index=False)
    return df_dict

def get_pickled_multiplicador():
    filename = os.path.join(os.path.dirname(__file__), '../pickles/multiplicador.pkl')
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            multiplicador = pkl.load(file)
    else:
        multiplicador = {domain: [(1, 0)] * len(DataClass.clusters) for domain in DataClass.domains}
    return multiplicador

def save_pickle_multiplicador(multipliers):
    filedir = os.path.join(os.path.dirname(__file__), '../pickles/multiplicador.pkl')
    with open(filedir, 'wb') as file:
        pkl.dump(multipliers, file)

def reset_multipliers():
    domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CMTS', 'CORE VOZ', 'ENTORNO',
               'TRANSPORTE - DX', 'TRANSPORTE - TX', 'TX - RADIOENLACES Y SATELITAL']
    df = pd.DataFrame(index=domains)
    df['1'], df['2'], df['3']  = 0, 0, 1
    df.to_csv(os.path.join(os.path.dirname(__file__), '../exportables/' + 'multipliers.csv'))
    st.success('Los multiplicadores volvieron a sus valores predeterminados.')


# Style Functions ------------------------------------------------------------------------------------------------------

def set_block_container_style(
    max_width: int = 1200,
    max_width_100_percent: bool = False,
    padding_top: int = 1,
    padding_right: int = 1,
    padding_left: int = 1,
    padding_bottom: int = 5,
    only_padding_top: bool = False
):

    if only_padding_top:
        st.markdown(
            f"""
        <style>
            .reportview-container .main .block-container{{
                padding-top: {1}rem
            }}
        </style>
        """,
            unsafe_allow_html=True,
        )
    else:
        COLOR = "black"
        BACKGROUND_COLOR = "#fff"
        if max_width_100_percent:
            max_width_str = f"max-width: 100%;"
        else:
            max_width_str = f"max-width: {max_width}px;"
        st.markdown(
            f"""
    <style>
        .reportview-container .main .block-container{{
            {max_width_str}
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
            unsafe_allow_html=True,
        )


def style_title(title: str):
    st.markdown(f"<h1 style='text-align: center; font-family:Graphik; color: #a100ff;'>{title}</h1>",
                unsafe_allow_html=True)

# Plot Functions--------------------------------------------------------------------------------------------------------
def show_history_stock_and_service_level_per_date(date_range: (), cram_selecter: []):
    # Stocks
    df_stock = pd.read_csv(r'../exportables/simulate_stock_teco.csv')
    df_stock.fecha = pd.to_datetime(df_stock.fecha)
    df_stock = df_stock[(df_stock.fecha >= date_range[0]) & (df_stock.fecha <= date_range[1])]
    df_stock = df_stock.groupby(["fecha", "cram"])["stock"].sum().reset_index()

    # Tickets
    df_tickets = pd.read_csv(r'../exportables/simulate_tickets_teco.csv')
    df_tickets.fecha = pd.to_datetime(df_tickets.fecha)
    df_tickets = df_tickets[(df_tickets.fecha >= date_range[0]) & (df_tickets.fecha <= date_range[1])]
    df_tickets = df_tickets.groupby(['fecha'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
    df_tickets.columns = ["fecha", 'suma', 'cantidad']

    # Initialize plot
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f'Evolución de Stock',
                                        f'Evolución de Nivel de Servicio Acumulado'),
                        vertical_spacing=0.1)

    # List of colors
    list_of_colors = px.colors.qualitative.Dark24
    dictionary_of_colors = {}
    for i, cram in enumerate(list(df_stock.cram.unique())):
        dictionary_of_colors[cram] = list_of_colors[i]

    # Stock plot
    if 'Stock Total' in cram_selecter:
        subset = df_stock.groupby(["fecha"])["stock"].sum().reset_index()
        subset.columns = ["fecha", "stock"]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='Stock Total',
                                 mode='lines',
                                 line_color='black'),
                      row=1, col=1)

    if 'Stock Total' in cram_selecter:
        cram_selecter.remove('Stock Total')
    for cram in cram_selecter:
        subset = df_stock[df_stock.cram == cram]
        fig.add_trace(go.Scatter(x=subset.fecha,
                                 y=subset.stock,
                                 name='<CRAM> ' + str(cram),
                                 mode='lines',
                                 line_color=dictionary_of_colors[cram]),
                      row=1, col=1)

    # Ticket plot
    fig.add_trace(go.Scatter(x=df_tickets.fecha,
                             y=df_tickets.suma.cumsum() / df_tickets.cantidad.cumsum(),
                             mode='lines',
                             name='Nivel de Servicio',
                             line_color='yellow'),
                  row=2, col=1)

    # Update Layout
    fig.update_layout(height=800, width=600)

    return fig

@st.cache(allow_output_mutation=True)
def plot_history():

    # Stocks
    my_path = os.path.join(os.path.dirname(__file__), '../exportables/')
    df_stock = pd.read_csv(f'{my_path}simulate_stock_teco.csv')
    df_stock.fecha = pd.to_datetime((df_stock.fecha))
    df_stock_cluster = df_stock.groupby(['fecha','cluster']).sum().reset_index()
    df_stock_cram = df_stock.groupby(["fecha", "cram"])["stock"].sum().reset_index()

    # Tickets
    df_tickets_00 = pd.read_csv(f'{my_path}/simulate_tickets_teco.csv')
    df_tickets_00.fecha = pd.to_datetime((df_tickets_00.fecha))
    df_tickets = df_tickets_00.groupby(['fecha', 'cram'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
    df_tickets.columns = ["fecha", 'cram', 'suma', 'cantidad']
    df_tickets_cluster = df_tickets_00.groupby(['fecha', 'cluster'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
    df_tickets_cluster.columns = ["fecha", 'cluster', 'suma', 'cantidad']

    # Initialize plot
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f'Evolución de Stock',
                                        f'Evolución de Nivel de Servicio Acumulado'),
                        vertical_spacing=0.1)

    # Plots---------------------------------------------------------------------------------------------------------
    # Stock plot
    subset = df_stock.groupby(["fecha"])["stock"].sum().reset_index()
    subset.columns = ["fecha", "stock"]
    fig.add_trace(go.Scatter(x=subset.fecha,
                             y=subset.stock,
                             name='Stock Total',
                             mode='lines',
                             line_color='blue',
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
    #Total
    subset = df_tickets_00.groupby(['fecha'])["Nivel_de_servicio"].agg(['sum', 'count']).reset_index()
    subset.columns = ["fecha", 'suma', 'cantidad']
    fig.add_trace(go.Scatter(x=subset.fecha,
                             y=subset.suma.cumsum() / subset.cantidad.cumsum(),
                             mode='lines',
                             name='Nivel de Servicio',
                             line_color='blue',
                             visible=False),
                  row=2, col=1)
    #list(df_tickets.cram.unique())

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
                                      True, True, True, True, True, True, True ,            # Stock CRAMs
                                      False,                                                # Total Service Level
                                      True, True, True, True, True, True, True,             # Service Level CRAMs
                                      False, False, False, False, False,                    # Stock cluster
                                      False, False, False, False, False]}],                 # Service Level cluster
                   label="<CRAM>")


    button2 = dict(method='update',
                   args=[{"visible": [False,
                                      False, False, False, False, False, False, False ,
                                      False,
                                      False, False, False, False, False, False, False,
                                      True, True, True, True, True,
                                      True, True, True, True, True]}],
                   label="<CLUSTER>")


    button3 = dict(method='update',
                   args=[{"visible": [True,
                                      False, False, False, False, False, False, False ,
                                      True,
                                      False, False, False, False, False, False, False,
                                      False, False, False, False, False,
                                      False, False, False, False, False]}],
                   label="<TOTAL>")

    # LayOut-------------------------------------------------------------------------------------------------------
    # Update Layout
    fig.update_layout(height=800, width=600,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=30)
                      )
    fig.update_layout(
        updatemenus=[dict(type='buttons',
                          buttons=[button1, button2, button3],
                          x=1.05,
                          xanchor="left",
                          y=0.2,
                          yanchor="top")])

    fig.update_yaxes(title_text="Stock", row=1, col=1)

    fig.update_xaxes(title_text="Mes",
                     row=2, col=1)
    fig.update_yaxes(title_text="Nivel de Servicio", row=2, col=1)

    return fig

@st.cache
def plot_redistribution_bar_plot(transit_stock_df):
    donadores = transit_stock_df.sum(axis=1)
    recibidores = transit_stock_df.sum(axis=0)
    fig = go.Figure(go.Bar(
        x=donadores,
        y=donadores.index,
        name='Cantidad Donada',
        text=donadores,
        texttemplate='%{text} u',
        textposition='auto',
        marker_color='#7D0FB1',
        orientation='h'))
    fig.add_trace(go.Bar(
        x=recibidores,
        y=recibidores.index,
        name='Cantidad Recibida',
        text=recibidores,
        texttemplate='%{text} u',
        textposition='auto',
        marker_color='#CC7CE8',
        orientation='h'))
    fig.update_layout(barmode='stack',
                      margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                      template= "plotly_white",
                      xaxis_titlefont_size=16,
                      yaxis_titlefont_size=16,
                      bargap=0.35,
                      height=800,
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=-0.3,
                          xanchor="right",
                          x=0.7
                      )
                      )
    fig.update_yaxes(title_text="GREP")

    fig.update_xaxes(title_text="Cantidad")

    return fig


@st.cache
def barchart_redistribution(s0, sf, is_filtered=False):
    domain_palette = ['#74299E', '#235785', '#7C1F48', '#B48121', '#5D6814', '#0F5A0F', '#818E19','#1818A8', '#0300A7']

    colors = {'TRANSPORTE - TX': domain_palette[0],
              'TRANSPORTE - DX': domain_palette[1],
              'TX - RADIOENLACES Y SATELITAL': domain_palette[2],
              'ACCESO - FIJA': domain_palette[3],
              'ACCESO - MOVIL': domain_palette[4],
              'CORE VOZ': domain_palette[5],
              'ENTORNO': domain_palette[6],
              'CMTS': domain_palette[7],
              'Other': domain_palette[8]}

    subset_s0 = s0.groupby(['Dominio']).sum()
    subset_sf = sf.groupby(['Dominio']).sum()

    fig = go.Figure()
    row_acum = np.array([0] * len(list(subset_s0.columns)))
    for row in subset_s0.iterrows():
        if is_filtered:
            color = '#172BB6'
            names = 'Stock Inicial'
            text = row[1]
            texttemplate = '%{text} u'
        else:
            color = colors[row[0]]
            names = row[0]
            text = None
            texttemplate = None
        fig.add_trace(go.Bar(x=subset_s0.columns,
                             y=row[1],
                             name=names,
                             marker_color=color,
                             base=row_acum,
                             offsetgroup=1,
                             text=text,
                             texttemplate=texttemplate,
                             textposition='auto',
                             ))
        row_acum += np.array(row[1])

    row_acum = np.array([0] * len(list(subset_s0.columns)))
    for row in subset_sf.iterrows():
        if is_filtered:
            color = '#852AA4'
            names = 'Stock Final'
            show_legend=True
            text = row[1]
            texttemplate = '%{text} u'
        else:
            color = colors[row[0]]
            names = row[0]
            show_legend = False
            text = None
            texttemplate = None
        fig.add_trace(go.Bar(x=subset_sf.columns,
                             y=row[1],
                             name=names,
                             marker_color=color,
                             showlegend=show_legend,
                             #marker_line_color='purple',
                             #marker_line_width=3,
                             offsetgroup=2,
                             base=row_acum,
                             text=text,
                             texttemplate=texttemplate,
                             textposition='auto',
                             ))
        row_acum += np.array(row[1])

    if is_filtered:
        annotation=None
        x=0.4
    else:
        annotation = [
            go.layout.Annotation(text=f'🢀 | 🢂', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=17,
                                     color='black'
                                 ), bgcolor="white",
                                 yref='paper', x=0.034, y=-0.53, bordercolor='red', borderwidth=0),
            go.layout.Annotation(text=f'Stock', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=14,
                                     color='black'
                                 ), bgcolor="white",
                                 yref='paper', x=0.01, y=-0.6, bordercolor='red', borderwidth=0),
            go.layout.Annotation(text=f'Inicial', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=14,
                                     color='black'
                                 ), bgcolor="white",
                                 yref='paper', x=0., y=-0.7, bordercolor='red', borderwidth=0),
            go.layout.Annotation(text=f'Stock', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=14,
                                     color='black'
                                 ), bgcolor="white",
                                 yref='paper', x=0.078, y=-0.6, bordercolor='red', borderwidth=0),
            go.layout.Annotation(text=f'Final', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=14,
                                     color='black'
                                 ), bgcolor="white",
                                 yref='paper', x=0.078, y=-0.7, bordercolor='red', borderwidth=0)
                    ]
        x = 0.15

    fig.update_layout(#barmode='group',
                      legend=dict(
                          x=x,
                          y=-0.7,
                          orientation="h",
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)'),
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=5  # top margin
                      ),
                      annotations=annotation,
                      height=300,
                      plot_bgcolor='white',
                      bargroupgap=0.02,
                      bargap=0.1
                      )
    fig.update_yaxes(title_text="Cantidad [u]")

    fig.update_xaxes(title_text="GREP")
    return fig

@st.cache
def barchart_purchases(purchase):
    subset_buyable = purchase[purchase['Tipo de Abastecimiento'].isin(['Buyable', 'SPM'])].groupby(
        ['Dominio'])['Presupuesto Compra', 'Cantidad Requerida'].sum().sort_values(by='Presupuesto Compra', ascending=False)
    subset = purchase.groupby(['Dominio'])['Presupuesto Compra', 'Cantidad Requerida'].sum().sort_values(by='Presupuesto Compra', ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(name='Cantidad [u]',
                   x=subset.index,
                   y=subset['Cantidad Requerida'],
                   yaxis='y',
                   text=subset['Cantidad Requerida'],
                   texttemplate='%{text} u',
                   textposition='auto',
                   offsetgroup=1,
                   ),
            #go.Bar(name='Presupuesto Total [US$]',
             #      x=subset.index,
              #     y=subset.Presupuesto,
               #    yaxis='y2',
                #   offsetgroup=2,
                 #  text=subset.Presupuesto,
                  # texttemplate='%{text:.2s}',
                   #textposition='auto',
                   #marker_color='#852AA4'),
            go.Bar(name='Presupuesto Comprables [US$]',
                   x=subset_buyable.index,
                   y=subset_buyable['Presupuesto Compra'],
                   yaxis='y2',
                   offsetgroup=2,
                   text=subset_buyable['Presupuesto Compra'],
                   texttemplate='%{text:.2s}',
                   textposition='auto',
                   marker_color='#852AA4')
        ],
        layout={
            'yaxis2': {'title': 'Presupuesto [USD]', 'overlaying': 'y', 'side': 'left'},
            'yaxis': {'title': 'Unidades', 'side': 'right'}
        }
    )

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group',
                      legend=dict(
                          x=0.35,
                          y=-0.8,
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)'),
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=5  # top margin
                      ),
                      height=300,
                      plot_bgcolor='white',
                      bargroupgap=0.02,
                      bargap=0.1
                      )
    return fig

@st.cache
def plot_redistribution_map(transit_stock_df, grep_type, is_filtered, format_map):

    lines_colors = {'low': 'seagreen',
                    'medium': 'orange',
                    'high': 'red',
                    'origin':'#2DB6DE',
                    'destiny':'blue'}

    df_greps = load_grep_geometry(grep_type)

    grep = {}
    for row in df_greps.iterrows():
        grep[row[0]] = {'lat': row[1][0], 'long': row[1][1]}

    fig = go.Figure(go.Scattermapbox())

    for index, row in df_greps.iterrows():
        fig.add_trace(go.Scattermapbox(
            lat=[row['lat']],
            lon=[row['long']],
            mode='markers',
            showlegend=False,
            marker=go.scattermapbox.Marker(
                size=15,
                color='blue',
                opacity=0.2,
            )
        ))

    rutes={}
    for i in range(0, len(list(transit_stock_df.index))):
        for j in range(i +1 , len(list(transit_stock_df.index))):
            rutes[(transit_stock_df.index[i] , transit_stock_df.index[j])] = int(transit_stock_df.iloc[i, j] + transit_stock_df.iloc[j, i])

    max_value = max(rutes.values())
    min_value = min(rutes.values())
    step = int(round((max_value - min_value)/3 , 0))

    for wh, amount in rutes.items():
        if amount > 0:
            if amount in range(min_value, min_value + step):
                color_line = lines_colors['low']
            elif amount in range(min_value + step, max_value - step +1):
                color_line = lines_colors['medium']
            else:
                color_line = lines_colors['high']
            if is_filtered:
                markers = {'size': 10, 'color': [lines_colors['origin'], lines_colors['destiny']]}
            else:
                markers = {'size': 10, 'color': [lines_colors['destiny'], lines_colors['destiny']]}
            fig.add_trace(go.Scattermapbox(
                mode="markers+lines",
                lon=[grep[wh[0]]['long'], grep[wh[1]]['long']],
                lat=[grep[wh[0]]['lat'], grep[wh[1]]['lat']],
                text=[wh[0], wh[1]],
                textposition="bottom right",
                marker=markers,
                showlegend=False,
                line={'color': color_line}))

    if is_filtered:
        annotation = [
            go.layout.Annotation(text=f'Entre {max_value - step} u - {max_value} u',  showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['high']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.5, bordercolor=lines_colors['high'], borderwidth=1),
            go.layout.Annotation(text=f'Entre {min_value} u - {min_value + step} u', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['low']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.72, bordercolor=lines_colors['low'], borderwidth=1),
            go.layout.Annotation(text=f'Entre {min_value + step} u - {max_value - step} u', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['medium']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.6, bordercolor=lines_colors['medium'], borderwidth=1),
            go.layout.Annotation(text=f'o Origen', showarrow=False,
                                 xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['origin']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.3, bordercolor=lines_colors['origin'], borderwidth=1),
            go.layout.Annotation(text=f'o Destino', showarrow=False,
                                 xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['destiny']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.2, bordercolor=lines_colors['destiny'], borderwidth=1)

        ]
    else:
        annotation = [
            go.layout.Annotation(text=f'Entre {max_value - step} u - {max_value} u', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['high']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.5, bordercolor=lines_colors['high'], borderwidth=1),
            go.layout.Annotation(text=f'Entre {min_value} u - {min_value + step} u', showarrow=False, xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['low']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.72, bordercolor=lines_colors['low'], borderwidth=1),
            go.layout.Annotation(text=f'Entre {min_value + step} u - {max_value - step} u', showarrow=False,
                                 xref='paper',
                                 font=dict(
                                     family="Courier New, monospace",
                                     size=16,
                                     color=lines_colors['medium']
                                 ), bgcolor="white",
                                 yref='paper', x=0.9, y=0.6, bordercolor=lines_colors['medium'], borderwidth=1)

        ]
    if format_map: style = 'open-street-map'
    else: style = 'carto-positron'
    fig.update_layout(
        margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
        showlegend=False,
        mapbox={
            'center': {'lon': -55, 'lat': -32},
            'style': style,  # open-street-map - carto-positron
            'zoom': 4},
        annotations=annotation)




    return fig

def plot_geus_info_price(state):


    st.markdown('**Distribución por red**')
    # SubPlot
    fig = go.Figure()

    #-------------------------------------------------------------------------------------------------------------------
    # Price Plot

    net_price = [(geu.domain, geu.price) for geu in state.geus_by_id.values()]
    networks = pd.DataFrame({'network': [i[0] for i in net_price],
                             'price': [i[1] for i in net_price]})

    for net in list(networks.network.unique()):
        fig.add_trace(go.Box(y=networks[networks.network == net].price,
                             name=str(net)))
    fig.update_layout(height=200,
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=5  # top margin
                      ))

    return fig

def plot_geus_info_procurement(state):

    st.markdown('**Tipos de GEUs**')
    df_procurement = pd.DataFrame({'comprable': [geu.is_buyable for geu in state.geus_by_id.values()],
                                   'reparable': [geu.is_repairable for geu in state.geus_by_id.values()],
                                   'desmontable': [geu.is_dismountable for geu in state.geus_by_id.values()],
                                   'spm': [geu.is_spm for geu in state.geus_by_id.values()]})

    df_procurement = df_procurement.groupby(['comprable','reparable','desmontable','spm']
                                            ).size().reset_index().rename(columns={0: 'count'})

    df_procurement["categoria"] = df_procurement.iloc[:,:-1].dot(df_procurement.columns[:-1] + ';').str.rstrip(';')

    fig = go.Figure(data=go.Scatter(
        x=[i for i in range(0, len(df_procurement))],
        y=df_procurement["categoria"],
        mode='markers+text',
        text=df_procurement["categoria"],
        marker=dict(size=df_procurement["count"],
                    sizeref = 5)
    ))
    fig.update_layout(height=400,
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=5  # top margin
                      ))

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )

    return fig

def plot_geus_info_pie(state):
    # Network Plot

    df_networks = pd.DataFrame({'network': [geu.domain for geu in state.geus_by_id.values()]})
    df_networks = df_networks.network.value_counts().reset_index()
    df_networks.columns = ["Red", "Cantidad"]

    fig = go.Figure(data=go.Pie(labels=list(df_networks["Red"].unique()),
                         values=list(df_networks["Cantidad"])))
    st.markdown('**Distribución por red**')
    fig.update_layout(height=150,
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=5  # top margin
                          ))
    fig.update_traces(hoverinfo='label+percent', textinfo='value')
    return fig

def example_map_plot():
    # -------------------
    import pydeck as pdk

    # GREAT_CIRCLE_LAYER_DATA = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/flights.json"  # noqa
    import json

    example = json.dumps([
        {
            "from": {
                "type": "major",
                "name": "Capital",
                "abbrev": "CABA",
                "coordinates": [
                    -58.4662931,
                    -34.5901061
                ]
            },
            "to": {
                "type": "major",
                "name": "Misiones",
                "abbrev": "MSN",
                "coordinates": [
                    -55.12172217,
                    -27.48480593
                ]
            }
        },
        {
            "from": {
                "type": "major",
                "name": "Capital",
                "abbrev": "CABA",
                "coordinates": [
                    -58.4662931,
                    -34.5901061
                ]
            },
            "to": {
                "type": "major",
                "name": "Concordia",
                "abbrev": "CNC",
                "coordinates": [
                    -58.01957958,
                    -31.39544675
                ]
            }
        },
        {
            "from": {
                "type": "major",
                "name": "Cordoba",
                "abbrev": "CBA",
                "coordinates": [
                    -64.29239426,
                    -36.62055941
                ]
            },
            "to": {
                "type": "major",
                "name": "Casa de Pato",
                "abbrev": "Pato",
                "coordinates": [
                    -58.5286,
                    -34.4708
                ]
            }
        }
    ])

    df = pd.read_json(example)

    # Use pandas to prepare data for tooltip
    df["from_name"] = df["from"].apply(lambda f: f["name"])
    df["to_name"] = df["to"].apply(lambda t: t["name"])

    # Define a layer to display on a map
    layer = pdk.Layer(
        "GreatCircleLayer",
        df,
        pickable=True,
        get_stroke_width=22,
        get_source_position="from.coordinates",
        get_target_position="to.coordinates",
        get_source_color=[64, 255, 0],
        get_target_color=[0, 128, 200],
        auto_highlight=True,
    )

    # Set the viewport location
    view_state = pdk.ViewState(latitude=-30, longitude=-55, zoom=4, bearing=0, pitch=0)

    # Render
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{from_name} to {to_name}"}, )
    r.picking_radius = 10

    return r


def show_demand_plot(geu, current_year):
    # Tickets
    ticket_demands = []
    for key, tickets in geu.tickets.items():
        ticket_demands.append([key, sum([ticket.amount for ticket in tickets])])
    df_tickets = pd.DataFrame(ticket_demands, columns=['date', 'ticketed'])
    df_tickets['date'] = pd.to_datetime(df_tickets['date']).dt.date
    df_tickets = df_tickets[df_tickets['date'].apply(lambda x: x.year) <= current_year]
    df_tickets = df_tickets[df_tickets['date'].apply(lambda x: x.year) >= current_year - 3]
    plot = plt.figure(f"Evolución Demanda - GEU {geu.id}")
    plt.title(f"Evolución Demandas - GEU {geu.id}", fontsize=18)
    plt.bar(df_tickets['date'], df_tickets['ticketed'], label='Tickets',
            width=timedelta(days=1), color='blue')
    plt.xticks(rotation=90)

    return plot


