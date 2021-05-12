import datetime as dt
from PIL import Image
import sys
import inspect
import os
from os.path import basename
from matplotlib import pyplot
from zipfile import ZipFile

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from front.st_classes import get_state
from front.st_functions import *
from front.st_validation import *
from front.st_simulate_teco import HistoryTeco
from validation import download_geus
from front.st_classes import Grid
import reinforcement_learning.optimizador_de_compras as opt
from reinforcement_learning.model_vs_history import ModelVSHistory


def st_main():
    # Initialize Session

    st.set_page_config(layout="centered", # Can be "centered" or "wide". In the future also "dashboard", etc.
                       initial_sidebar_state="auto", # Can be "auto", "expanded", "collapsed"
                       page_title='Neural Teco', # String or None. Strings get appended with "• Streamlit".
                       page_icon=Image.open(r'Images/acc_icon.png'))  # String, anything supported by st.image, or None.

    state = get_state()

    if state.is_first_state is None:
        clean_state(state)

    image = Image.open(r'Images/neuronal_teco.png')
    st.sidebar.image(image, use_column_width=True)

    pages = {"Menú Principal": page_menu,
             "Cargar Nuevos Datos": page_update_data,
             "Optimizador de Stock de Seguridad": optimizer,
             "Ejecución del Modelo": page_run_main_process,
             "Materiales / GEUs": page_geus,
             "Requerimientos": page_purchase,
             "Redistribución": page_redistribution,
             "Dashboard": dashboard
             }

    page = st.sidebar.selectbox("Selecciona una página", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page](state)

def page_menu(state):
    # Title
    # _set_block_container_style()
    style_title('Welcome to Neural Teco')

    set_block_container_style(only_padding_top=True)

    st.write('\n')
    st.markdown("*Bienvenidos a Supply Brain by Accenture (v1.0.0)*")

    st.write("Esta herramienta consta de 6 módulos:")
    st.markdown(f":clipboard: **Cargar Nuevos Datos**: Permite visualizar la estructura de la información que se encuentra "
                f"actualmente en la base de datos, además de poder subir versiones actualizadas de estos archivos. "
                f"Los datos deben procesarse para poder desbloquear los siguientes módulos.")
    st.markdown(f":brain: **Optimizador de Stock de Seguridad**: Sección destinada a la configuración del stock de seguridad. "
                f"Contiene un optimizador de dicho valor en función de un dominio, cluster y Disponibilidad de Repuesto. También permite la carga manual.")
    st.markdown(f":arrow_forward: **Ejecución del Modelo**: Selección de uno de los modelos pre-entrenados."
                f"Este módulo debe ser ejecutado para poder visualizar eseleccil resto de los módulos.")
    st.markdown(f":loop: **Materiales / GEUs**: Panel de visualización de Materiales o GEUs. Incluye imagen con sus equivalencias internas, "
                f"características propias del material y gráficos representativos de los datos.")
    st.markdown(f":heavy_dollar_sign: **Requerimientos**: Información tabulada de la siguiente compra propuesta por la nueva "
                f"política. Puede verse a nivel general, a nivel GEU e incluso bajar la información a un archivo CSV.")
    st.markdown(f":truck: **Redistribución**: Información tabulada de la redistribución de materiales "
                f"propuesta por la nueva política. Puede verse a nivel general, a nivel GEU, a nivel centro e incluso "
                f"bajar la información a un archivo CSV.")
    #st.markdown(f":chart_with_upwards_trend: **Resultados**: Módulo estandarizado para la medición y comparación de "
    #            f"resultados de la política histórica vs. la nueva política simulada.")

    st.markdown(get_binary_file_downloader_html(r"../exportables/Manual_de_Usuario.pdf", 'Manual de Usuario'), unsafe_allow_html=True)

    end_page()

def page_update_data(state):
    # Set page config
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)

    # Title
    style_title('Cargar Nuevos Datos')
    st.markdown("*Sección destinada a la actualización de los datos de entrada al modelo (si aplica). "
                "El formato de los nuevos datos debe ser excel. "
                "Una vez cargado el archivo se podrá selecionar entre dos opciones:*")
    st.markdown("*- Cargar Nuevos Datos: cuando el archivo contenga solamente nuevos registros de datos*")
    st.markdown("*- Borrar y Recargar Datos: en caso de necesidad de eliminar los viejos datos y reemplazarlos "
                "con el nuevo archivo cargado*")

    # SQL Tables types
    info = {'Materiales': {'sql_name': 'produccion.materiales',
                          'columns_types':['String' for _ in range(9)] + ['Integer', 'String', 'Integer', 'String', 'String'] + ['Bool' for _ in range(4)] + ['Integer', 'Integer']},
            'Demandas': {'sql_name': 'produccion.tickets_s4',
                        'columns_types': ['String', 'String', 'Integer', 'String']},
            'Stock': {'sql_name': 'produccion.stock_s4',
                      'columns_types': ['String', 'Date', 'Integer', 'String']},
             #'Leadtimes (OEs)': {'sql_name': 'p01_oes',
             #                    'columns_types':['Integer', 'Integer', 'Integer', 'Date', 'Date', 'String', 'Integer']},
             #'Movimientos': {'sql_name': 'r00_total_historic_movements',
             #                'columns_types': ['String' for _ in range(30)]},
            'Relación GREP-Almacén': {'sql_name': 'produccion.relacion_grep_almacen',
                                    'columns_types': ['String', 'String', 'String', 'String', 'String', 'String', 'String',
                                                 'String', 'String', 'Integer', 'Integer', 'String', 'String', 'String']},
            'GREPs Habilitados': {'sql_name': 'produccion.greps_habilitados',
                                  'columns_types': ['String']},
            'Validación GEU-Dominio': {'sql_name': 'produccion.validacion_geu_dominio',
                                   'columns_types': ['String', 'String', 'String']},
            'Renombrar GEU': {'sql_name': 'produccion.geu_name_override',
                                   'columns_types': ['String', 'String']},
            'Leadtimes - Reparables':{'sql_name': 'produccion.leadtimes_reparables',
                                   'columns_types': ['String', 'Integer']},
            'Leadtimes - SPM': {'sql_name': 'produccion.leadtimes_spm',
                                       'columns_types': ['String', 'String', 'Integer']},
            'Equivalencias': {'sql_name': 'produccion.equivalencias',
                              'columns_types': ['String' for _ in range(2)] + ['Integer'] + ['String' for _ in range(7)]},
            'OES': {'sql_name': 'produccion.oes_s4',
                    'columns_types': ['Integer', 'Integer', 'Integer', 'Date', 'Date', 'String', 'Integer', 'String']}
             }

    for type, options in info.items():
        show_update_options(name=type, sql_name= options['sql_name'], types=options['columns_types'])

    end_page()


def optimizer(state):
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)
    # show_cleaning_button(state)
    style_title("Optimizador de Stock de Seguridad")


    # Manual Input
    manual_input = st.sidebar.checkbox('Colocar SS Manualmente')

    clusters = {'Todos': 'all',
                'Cluster 1 - Demanda Anual hasta 2 unidades': 1,
                'Cluster 2 - Demanda Anual hasta 12 unidades': 2,
                'Cluster 3 - Demanda anual mayor a 12 unidades': 3}
    domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
               'TX - RADIOENLACES Y SATELITAL']

    grep_types = {'Único GREP': 'unicram', '7 GREPs (Histórico)': '7_crams', 'GREPs': 'greps'}

    if manual_input:
        multipliers = opt.get_multipliers(is_streamlit=True, resupplies_per_year=None, grep_type=None)
        domain = st.selectbox("Ingrese Dominio", domains)
        cluster = clusters[st.selectbox("Ingrese Cluster", list(clusters.keys())[1:], index=0)]
        state.cluster = int(cluster)
        state.ss = st.number_input('Stock de Seguridad',
                                       min_value=0,
                                       value=2,
                                       step=1)
        if state.cluster == 3:
            state.buy = st.selectbox('Porcentaje de Compras',
                                       options=[str(x)+'%' for x in range(50,110,10)],
                                       index=5)
            state.buy = int(state.buy[:-1])/100

        if st.button('<Guardar Cambios>'):
            multipliers[domain][cluster - 1] = (state.buy, state.ss)
            opt.save_multipliers(multiplicadores=multipliers, cluster=state.cluster, is_streamlit=True, resupplies_per_year=None, grep_type=None)
    else:
        st.markdown('*Se considera como Stock de Seguridad al total de la red, **NO** por grep. Se parte de un stock de 0 unidades.*')

        cols = st.beta_columns(2)

        list_of_stock = load_stock_dates()
        stock_date = cols[1].selectbox('Fecha de Inicio de simulación', options=list_of_stock[:-1], index=0)

        cram_grep_option = grep_types[cols[0].selectbox('Tipo de Topología',
                                                   options=list(grep_types.keys()),
                                                   index=2)]
        sls = {f'{x}%': x for x in range(10, 110, 10)}
        domain = cols[0].selectbox("Ingrese Dominio", domains)
        cluster = clusters[cols[0].selectbox("Ingrese Cluster", list(clusters.keys()), index=1)]
        rep_per_year = int(cols[1].selectbox("Compras por año:", [1,2,3,4], index=0))

        if cluster == 'all':
            state.cluster = None
        else:
            state.cluster = int(cluster)

        sl = cols[1].selectbox("Ingrese Disponibilidad de Repuesto", list(sls.keys()), index=4)

        cols[1].markdown('**:floppy_disk: No Olvidarse :floppy_disk:**')
        if cols[1].button('<Guardar Cambios>'):
            if state.button_save: opt.save_multipliers(multiplicadores=state.multipliers, cluster=state.cluster,
                                                       resupplies_per_year=None, grep_type=None,is_streamlit=True)
            state.button_save = False
            state.multipliers = None
            state.cluster = None

        cols[0].markdown('**:brain: Iniciar Optimización**')
        if cols[0].button('<Optimizar>'):
            end_date = load_last_date_stock().iloc[0, 0]
            # st.markdown(f'*Fecha Final de Simulación: {end_date}*')
            state.multipliers = opt.optimizar(int(sl[:-1]) / 100, cluster=state.cluster, domain=domain, grep_type=cram_grep_option,
                                              chosen_heuristic="Action1", step=(.1, 1), is_streamlit=True,
                                              start_date=dt.datetime(stock_date.year,stock_date.month,stock_date.day),
                                              end_date=dt.datetime(end_date.year,end_date.month,end_date.day),
                                              resupplies_per_year=rep_per_year)
            state.button_save = True

    # Side bar------------------------
        if rep_per_year == 1: rep=1
        else: rep=4
        if cram_grep_option=='unicram': type='7_crams'
        else: type=cram_grep_option
        if st.sidebar.button('Stock de Seguridad Optimizando por Disponibilidad de Repuesto'):
            set_multiplier_to_deafult(type='sl', resupplies_per_year=rep, grep_type=type)

        if st.sidebar.button('Stock de Seguridad Optimizando por Costo'):
            set_multiplier_to_deafult(type='low', resupplies_per_year=rep, grep_type=type)

        if st.sidebar.button('Llevar Valores Stock de Seguridad a 0'):
            opt.reset_multipliers(is_streamlit=True, resupplies_per_year=None, grep_type=None)

    multipliers = opt.get_multipliers(is_streamlit=True, resupplies_per_year=None, grep_type=None)

    domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
               'TX - RADIOENLACES Y SATELITAL']

    df_multipliers = pd.DataFrame(index=domains, columns=[str(i) for i in DataClass.clusters])
    for dom, list_cluster in multipliers.items():
        for cluster, tupla in enumerate(list_cluster):
            if tupla[1] < 0 : value = 0
            else: value = tupla[1]
            if cluster + 1 != 3:
                df_multipliers.at[dom, str(cluster + 1)] = str(value) + ' u'
            else:
                df_multipliers.at[dom, str(cluster + 1)] = '(' + str(round(tupla[0] * 100,2)) + '%) ' + str(
                    value) + ' u'

    st.sidebar.table(df_multipliers)
    # End side Bar-------------------

def page_run_main_process(state):
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)

    # Title
    style_title("Ejecución del Modelo")

    process_finish = st.empty()

    st.markdown('**Configuración General**')

    cols = st.beta_columns(4)
    days_between_movements = cols[2].selectbox('Días entre redistribuciones',
                                       options=[15,30,45,60],
                                       index=0)

    purchases_per_year = cols[1].selectbox('Compras por Año',
                                      options=[1, 2, 3, 4],
                                      index=0)

    grep_types = {'Único GREP': 'unicram', '7 GREPs (Histórico)': '7_crams', 'GREPs': 'greps'}
    state.cram_grep_option = grep_types[cols[0].selectbox('Tipo de Topología',
                                               options=list(grep_types.keys()),
                                               index=2)]

    state.is_greps_available = st.checkbox('Distribuir cluster 1 (baja demanda) entre GREPs habilitados', value=True)

    if purchases_per_year == 1:rep = purchases_per_year
    else:rep = 4
    if state.cram_grep_option == 'unicram':type = '7_crams'
    else:type = state.cram_grep_option

    if st.sidebar.button('Stock de Seguridad Optimizando por Disponibilidad de Repuesto'):
        set_multiplier_to_deafult(type='sl', resupplies_per_year=rep, grep_type=type)

    if st.sidebar.button('Stock de Seguridad Optimizando por Costo'):
        set_multiplier_to_deafult(type='low', resupplies_per_year=rep, grep_type=type)

    if st.sidebar.button('Llevar Valores Stock de Seguridad a 0'):
        opt.reset_multipliers(is_streamlit=True, resupplies_per_year=None, grep_type=None)

    #date = st.date_input('Fecha de simulación', value=dt.date.today(), min_value=dt.datetime(2018, 1, 1), max_value=None)
    #date = load_last_date_stock()
    #state.date = date.iloc[0,0]
    list_of_stock = load_stock_dates()
    state.date = cols[3].selectbox('Fecha de Stock', options=sorted(list_of_stock,reverse=True)[:5], index=0)
    #st.markdown(f'**Última Fecha de Stock:** {state.date}')
    #st.markdown('*Para cambiar la fecha de stock es necesario cargar nuevos datos de stock en la sección de "Cargar Nuevos Datos"*')

    st.markdown('**Stock de Seguridad**')

    multipliers = opt.get_multipliers(is_streamlit=True, resupplies_per_year=None, grep_type=None)

    domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
               'TX - RADIOENLACES Y SATELITAL']

    df_multipliers = pd.DataFrame(index=domains, columns=['Cluster ' + str(i) for i in DataClass.clusters])
    for dom, list_cluster in multipliers.items():
        for cluster, tupla in enumerate(list_cluster):
            if tupla[1] < 0: value = 0
            else: value = tupla[1]
            if cluster + 1 != 3:
                df_multipliers.at[dom, 'Cluster ' + str(cluster + 1)] = str(value) + ' u'
            else:
                df_multipliers.at[dom, 'Cluster ' + str(cluster + 1)] = '(' + str(round(tupla[0]*100,2)) + '%) ' + str(value) + ' u'

    st.markdown("###### *【Cluster 1: demanda < 2 u/año】 - "
                "【Cluster 2: demanda 2 - 12 u/año】 - "
                "【Cluster 3: demanda > 12 u/año】*")
    st.table(df_multipliers)
    st.markdown('*El stock se seguridad se puede modificar manualmenteo u optimizado en la sección "Optimizador de Stock de Seguridad"*')

    st.markdown('**Comenzar carga de datos**')
    process_data = st.button("<Iniciar>")

    # Initialize bar progress
    bar_progress = st.progress(0)
    # Process Data
    if process_data:
        st.header("Welcome to Neural Teco")
        state.env_config = {"days_between_movements": days_between_movements,
                      "resupplies_per_year": purchases_per_year,
                      "use_historic_stocks": True,
                      "simulate_purchases": True,
                      'grep_type': state.cram_grep_option,
                      'from_pickle': False,
                      'is_streamlit': True,
                      'bar_progress': bar_progress,
                      'print_on':False,
                      "start_date": dt.datetime(state.date.year, state.date.month, state.date.day),
                      #'start_date': dt.datetime(2018,7,1),
                      "end_date": dt.datetime(2020, 1, 1),
                      "multiplicador": opt.get_multipliers(is_streamlit=True,resupplies_per_year=None, grep_type=None),
                      "is_s4": True
                    }

        state.env = SimulationEnvironment(state.env_config)
        state.DataClass = state.env.data_class
        state.geus_by_id = state.env.data_class.geus_by_id
        state.has_purchase = False
        state.distributions = {}
        process_finish.success("Carga de Datos Finalizada!")

    end_page()


def page_geus(state):
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)

    # Title
    style_title("Información General")

    if not state.geus_by_id:
        st.markdown('*La sección \"Ejecución\" debe ser utilizada antes de visualizar los GEUs.*')

        image = Image.open(r'Images/not_geus.jpg')
        st.image(image, use_column_width=False)
    else:
        # GEUS checkbox
        select_geu = st.sidebar.checkbox('Seleccionar GEU')

        if st.sidebar.checkbox('Descargar Diccionario de GEUs completo'):
            show_dowload_option(generate_geu_dictionary(state), 'diccionario_de_geus', is_sidebar=True)

        if st.sidebar.checkbox('Descargar Diccionario Material-GEU'):
            show_dowload_option(generate_mat_geu_dictionary(state), 'relacion_material_geu', is_sidebar=True)

        if st.sidebar.checkbox('Descargar Imágenes de GEUs'):
            download_geus(geus=state.geus_by_id, include_unigeus=False, print_on=False)

            # create a ZipFile object
            with ZipFile(r'../exportables/imagenes_geus.zip', 'w') as zipObj:
                # Iterate over all the files in directory
                for folderName, subfolders, filenames in os.walk(r"../exportables/GEUs"):
                    for filename in filenames:
                        # create complete filepath of file in directory
                        filePath = os.path.join(folderName, filename)
                        # Add file to zip
                        zipObj.write(filePath, basename(filePath))

            st.sidebar.markdown(get_binary_file_downloader_html(r"../exportables/imagenes_geus.zip", 'Imagenes_de_geus'),
                        unsafe_allow_html=True)


        if select_geu:
            # Selector
            geu_id = st.sidebar.selectbox('Seleccionar GEU, Nombre o Material',
                                              options=list(state.geus_by_id.keys()) + list(set(
                                                  list(state.geus_by_id.keys()) + [x.name for x in
                                                                                   list(state.geus_by_id.values())] + list(
                                                      state.DataClass.materials_by_id.keys())) - set(
                                                  list(state.geus_by_id.keys()))))

            if geu_id:
                if geu_id in state.DataClass.materials_by_id:
                    geu_id = state.DataClass.materials_by_id[geu_id].geu.id
                for it in [item.id for item in state.DataClass.geus_by_id.values() if geu_id == item.name]:
                    geu_id = it
                    break
                try:
                    state.geus_by_id[geu_id].get_info(is_streamlit=True)
                    state.geus_by_id[geu_id].plot_bare_graph(show_plot=False)
                except:
                    st.warning('El código de GEU ingresado no fue encontrado en la base. '
                               'La lista de GEUs válida es la siguiente:')
                    st.write(state.geus_by_id.keys())
                fig = pyplot.gcf()
                fig.set_size_inches(5, 4)
                st.pyplot(fig)
                fig.clf()
                st.pyplot(fig=show_demand_plot(geu=state.geus_by_id[geu_id],
                                               current_year=state.date.year)
                          , use_container_width=True)
        else:
            amount = len(list(state.geus_by_id.values()))
            total_materials = len(list(state.DataClass.materials_by_id.keys()))
            domains = [material.domain for material in state.DataClass.materials_by_id.values()]
            cols = st.beta_columns(3)
            cols[0].markdown(f'**Total de GEUs:** {amount}')
            cols[1].markdown(f'**Total de Materiales**: {total_materials}')
            cols[2].markdown(f'**Total de Redes:** {len(set(domains))}')

            #st.plotly_chart(plot_geus_info_pie(state), use_container_width=True)
            #st.plotly_chart(plot_geus_info_price(state), use_container_width=True)
            # Procurement plot
            #st.plotly_chart(plot_geus_info_procurement(state), use_container_width=True)

            # Load charts
            if state.is_first_geu_info:
                data = validation_data(state.geus_by_id)
                state.figures = validation(data)
                state.is_first_geu_info = False

            plot_selected = st.selectbox(label='Seleccioná un gráfico', options=list(state.figures.keys()), index=0)
            st.pyplot(fig=state.figures[plot_selected], use_container_width=True)

    end_page()


def page_purchase(state):
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)

    set_block_container_style()

    # Title
    style_title("Requerimientos")

    if not state.geus_by_id:
        st.markdown('*La sección \"Ejecución\" debe ser utilizada antes de visualizar los GEUs.*')

        image = Image.open(r'Images/not_geus.jpg')
        st.image(image, use_column_width=False)
    else:
        if not state.has_purchase:
            progress_bar = st.progress(0)

            total_geus = len(list(state.geus_by_id.values()))

            env = state.env

            # Variables for Table----------------
            geus_id = []
            name = []
            materials = []
            amount = []
            unit_price = []
            total_price = []
            leadtimes = []
            domain = []
            crit = []
            type = []
            stock = []
            cluster = []
            forecasts = []
            req = []
            marca = []
            equipo = []
            buyable = []
            spm = []
            repairable = []
            dismountable =[]
            price_unit = []
            rep_price = []
            rep_unit = []
            total_rep = []

            #----------------------------------
            i = 0
            for geu_id in state.geus_by_id.keys():
                i += 1
                if state.geus_by_id[geu_id].cluster == 0:
                    sim_amount = 0
                    forecast = 0
                    env.geu = env.data_class.geus_by_id[geu_id]
                    env.reset()
                else:
                    env.geu = env.data_class.geus_by_id[geu_id]
                    env.reset()
                    sim_amount, leadtime, forecast = env.sim_purchases(get_purchase=True)

                #if state.geus_by_id[geu_id].procurement_type == 'Repairable': multi = 0.4
                #else: multi=1

                geus_id.append(geu_id)
                name.append(state.geus_by_id[geu_id].name)
                materials.append([material.catalog for material in state.geus_by_id[geu_id].materials])
                amount.append(int(sim_amount))
                unit_price.append(state.geus_by_id[geu_id].price)
                total_price.append(int(sim_amount)*int(int(state.geus_by_id[geu_id].price)))
                total_rep.append(int(sim_amount)*int(int(state.geus_by_id[geu_id].rep_price))
                                 if state.geus_by_id[geu_id].is_repairable else 0)
                leadtimes.append(str(state.geus_by_id[geu_id].leadtime) + ' +/- ' + str(state.geus_by_id[geu_id].leadtime_sd))
                domain.append(state.geus_by_id[geu_id].domain)
                crit.append(state.geus_by_id[geu_id].criticality)
                type.append(state.geus_by_id[geu_id].procurement_type)
                stock.append(sum(env.stock_by_wh))
                cluster.append(state.geus_by_id[geu_id].cluster)
                forecasts.append(round(float(forecast),2))
                req.append('anual' if state.env_config["resupplies_per_year"]==1 else
                           'semestral' if state.env_config["resupplies_per_year"]==2 else
                           'cuatrimestral' if state.env_config["resupplies_per_year"]==3 else 'trimestral')
                marca.append(state.geus_by_id[geu_id].brand)
                equipo.append(state.geus_by_id[geu_id].equipment)
                buyable.append(state.geus_by_id[geu_id].is_buyable)
                spm.append(state.geus_by_id[geu_id].is_spm)
                repairable.append(state.geus_by_id[geu_id].is_repairable)
                dismountable.append(state.geus_by_id[geu_id].is_dismountable)
                price_unit.append(state.geus_by_id[geu_id].price_unit)
                rep_price.append(state.geus_by_id[geu_id].rep_price if state.geus_by_id[geu_id].is_repairable else
                                 0)
                rep_unit.append(state.geus_by_id[geu_id].rep_unit)

                progress_bar.progress(i / total_geus)

            progress_bar.progress(100)

            purchase = pd.DataFrame({'GEU':geus_id,
                                     'Descripcion':name,
                                     'Catálogos':materials,
                                     'Cantidad Requerida':amount,
                                     'Tipo Requerimiento':req,
                                     'Precio Unitario Compra':unit_price,
                                     'Unidad Precio':price_unit,
                                     'Presupuesto Compra':total_price,
                                     'Preio Unitario Reparacion':rep_price,
                                     'Unidad Reparación':rep_unit,
                                     'Presupuesto Reparacion':total_rep,
                                     'Lead Time':leadtimes,
                                     'Dominio':domain,
                                     'Marca':marca,
                                     'Equipo':equipo,
                                     'Criticidad':crit,
                                     'Stock':stock,
                                     'Tipo de Abastecimiento':type,
                                     'Cluster':cluster,
                                     'Forecast': forecasts,
                                     'Comprable': buyable,
                                     'SPM': spm,
                                     'Reparable': repairable,
                                     'Desmontable': dismountable})
            purchase.Cluster = purchase.Cluster.apply(lambda x: str(x)+' (baja)' if x == 1 else str(x)+' (media)'
            if x==2 else str(x)+' (nula)' if x==0 else str(x)+' (alta)')
            state.purchase = purchase
            state.has_purchase = True

        # Filters-------------------------------------------------------------------------------------------------------
        # Multi Selector
        geus_ids = st.sidebar.multiselect('Seleccionar GEU, Nombre o Material', options=list(state.geus_by_id.keys()) + list(set(list(state.geus_by_id.keys()) + [x.name for x in list(state.geus_by_id.values())] + list(state.DataClass.materials_by_id.keys())) - set(list(state.geus_by_id.keys()))))

        # 0 Amount
        zero = None
        if st.sidebar.checkbox('Filtrar con Cantidad Requerida', value=True):
            zero = True

        # Domain
        domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
                   'TX - RADIOENLACES Y SATELITAL']
        domain = None
        if st.sidebar.checkbox('Filtrar por Dominio'):
            domain = st.sidebar.selectbox('Seleccionar Dominio', options=domains)

        # Cluster
        cluster = None
        if st.sidebar.checkbox('Filtrar por Cluster'):
            cluster = st.sidebar.selectbox('Seleccionar Cluster', options=['0 (nula)', '1 (baja)','2 (media)','3 (alta)'], index=1)

        # Criticality
        crit = None
        if st.sidebar.checkbox('Filtrar por Criticidad'):
            crit = st.sidebar.selectbox('Seleccionar Criticidad', options=['critico','mayor','bajo'])

        # procurement_types
        procurement_types = ['SPM', 'Buyable', 'Dismountable', 'Repairable']
        type = None
        if st.sidebar.checkbox('Filtrar por Abastecimiento'):
            type = st.sidebar.selectbox('Seleccionar Abastecimiento', options=procurement_types)

        # Amount
        amount = None
        if st.sidebar.checkbox('Filtrar por Cantidad'):
            min, max = int(state.purchase['Cantidad Requerida'].min()), int(state.purchase['Cantidad Requerida'].max())
            amount = st.sidebar.slider('Cantidad Mínimo',
                                       min_value=min,
                                       max_value=max,
                                       step = 1)
        # procurement_types
        req_types = ['anual', 'mensual']
        req = None
        if st.sidebar.checkbox('Filtrar por Tipo de Requerimiento'):
            req = st.sidebar.selectbox('Seleccionar Abastecimiento', options=req_types)

        # Budget
        budget = None
        if st.sidebar.checkbox('Filtrar por Presupuesto'):
            min, max = int(state.purchase['Presupuesto Compra'].min()), int(state.purchase['Presupuesto Compra'].max())
            budget = st.sidebar.slider('Presupuesto Mínimo',
                                       min_value=min,
                                       max_value=max,
                                       step = int(round(max/min,0)))

        # Stock
        stock = None
        if st.sidebar.checkbox('Filtrar por Stock'):
            min, max = int(state.purchase['Stock'].min()), int(state.purchase['Stock'].max())
            stock = st.sidebar.slider('Stock Mínimo',
                                       min_value=min,
                                       max_value=max,
                                       step = 1)

        # Price
        price = None
        if st.sidebar.checkbox('Filtrar por Precio Unitario'):
            min, max = int(state.purchase['Precio Unitario Compra'].min()), int(state.purchase['Precio Unitario Compra'].max())
            price = st.sidebar.slider('Precio Mínimo',
                                       min_value=min,
                                       max_value=max,
                                       step = int(round(max/min,0)))

        has_purchase = False
        is_filtered = False
        if geus_ids or domain or cluster or type or crit or budget or amount or price or stock or zero or req:
            is_filtered = False
            if domain or cluster or type or crit or budget or amount or price or stock or zero:
                purchase = state.purchase
                if domain:
                    purchase = purchase[purchase['Dominio'] == domain]
                if cluster:
                    purchase = purchase[purchase['Cluster'] == cluster]
                if type:
                    purchase = purchase[purchase['Tipo de Abastecimiento'] == type]
                if crit:
                    purchase = purchase[purchase['Criticidad'] == crit]
                if budget:
                    purchase = purchase[purchase['Presupuesto Compra'] >= budget]
                if price:
                    purchase = purchase[purchase['Precio Unitario Compra'] >= price]
                if amount:
                    purchase = purchase[purchase['Cantidad Requerida'] >= int(amount)]
                if stock:
                    purchase = purchase[purchase['Stock'] >= stock]
                if zero:
                    purchase = purchase[purchase['Cantidad Requerida'] > 0]
                if req:
                    purchase = purchase[purchase['Tipo de Abastecimiento'] == req]
                has_purchase = True
            if geus_ids:
                for multi in geus_ids:
                    if multi in state.DataClass.materials_by_id:
                        geus_ids[geus_ids.index(multi)] = state.DataClass.materials_by_id[multi].geu.id
                purchase = state.purchase[state.purchase.GEU.isin(geus_ids) | state.purchase.Descripción.isin(geus_ids)]
                has_purchase = True

        else:
            purchase = state.purchase
            has_purchase = state.has_purchase
        # End filters---------------------------------------------------------------------------------------------------

        if has_purchase:
            st.markdown('## Resumen')
            #num = purchase.Presupuesto.sum()
            #st.markdown(f'**Presupuesto Total:** US$ {num:,}')
            if purchase['Cantidad Requerida'].sum() == 0:
                if (purchase['Cluster'] == '0 (nula)').sum() > 0:
                    st.info('El Material pertenece al cluster 0 (sin demandas), No requiere compra')
                else:
                    st.info('El Material no requiere compra')
                stock = purchase.Stock.sum()
                if stock == 1: st.markdown(f'**Stock a la fecha:** {stock} unidad *({state.date})*')
                if stock != 1: st.markdown(f'**Stock a la fecha:** {stock} unidades *({state.date})*')
            else:
                buyable_budget = purchase[purchase['Tipo de Abastecimiento'].isin(['Buyable','SPM'])]['Presupuesto Compra'].sum()
                spm_budget = purchase[purchase['Tipo de Abastecimiento'].isin(['SPM'])]['Presupuesto Compra'].sum()
                st.markdown(f"**Presupuesto Comprables:** US$ {buyable_budget:,} *(Comprables: US$ {(buyable_budget-spm_budget):,} y SPM: US$ { spm_budget:,})*")
                units_b = purchase[purchase['Tipo de Abastecimiento'].isin(['Buyable','SPM'])]['Cantidad Requerida'].sum()
                unit_spm = purchase[purchase['Tipo de Abastecimiento'].isin(['SPM'])]['Cantidad Requerida'].sum()
                units_r = purchase['Cantidad Requerida'].sum()
                units_rep = purchase[purchase['Tipo de Abastecimiento'].isin(['Repairable'])]['Cantidad Requerida'].sum()
                st.markdown(f'**Materiales a Comprar:** {units_b} unidad/es *(Comprables: {units_b-unit_spm} unidad/es y SPM: {unit_spm} unidad/es)*')
                st.markdown(f'**Materiales a Reparar/Desmontar:** {units_r-units_b} unidad/es *(Reparables: {units_rep} unidad/es y Desmontables: {units_r-units_b-units_rep} unidad/es)*')
                total_geus = len(purchase.GEU.unique())
                geus_c = len(purchase[purchase['Tipo de Abastecimiento'].isin(['Buyable', 'SPM'])].GEU.unique())
                st.markdown(f'**GEUS Totales:** {total_geus} *({geus_c} comprables y {total_geus-geus_c} reparables/desmontables)*')
                stock = purchase.Stock.sum()
                if stock == 1: st.markdown(f'**Stock a la fecha:** {stock} unidad *({state.date})*')
                if stock != 1: st.markdown(f'**Stock a la fecha:** {stock} unidades *({state.date})*')

                st.markdown(f"**Distribución de Presupuesto**")
                st.plotly_chart(barchart_purchases(purchase), use_container_width=True)

            st.markdown('## Requerimientos Propuestos')
            st.markdown('###### *Se muestran los primeros 10 registros de la tabla de resultados. El presupuesto estimado se enceuntra expresado a valor de compra. '
                        'Para verla completa descargar el archivo* ')
            st.table(purchase.head(10).style.format({"Cantidad Requerida": "{} u",
                                                     "Stock": "{} u",
                                                     "Precio Unitario": "{:20,.0f}",
                                                     "Precio Reparacion": "{:20,.0f}",
                                                     "Forecast": "{:20,.2f} u",
                                                     "Presupuesto": "USD{:20,.0f}"}))
            purchase = state.purchase[state.purchase['Cantidad Requerida'] > 0]
            show_dowload_option(purchase, f'requerimientos_{state.date}')

    end_page()


def page_redistribution(state):
    set_block_container_style(only_padding_top=True)

    #show_cleaning_button(state)

    # Title
    set_block_container_style()

    style_title('Redistribución')

    if not state.geus_by_id:
        st.markdown('*La sección \"Ejecución\" debe ser utilizada antes de visualizar los GEUs.*')

        image = Image.open(r'Images/not_geus.jpg')
        st.image(image, use_column_width=False)
    else:
        if state.env_config['grep_type'] == 'unicram':
            st.warning('No se requieren redistribuciones para un único GREP')
        else:
            if not state.distributions:

                if state.is_greps_available and state.cram_grep_option=='greps':
                    greps_availables = create_query("""select * from produccion.greps_habilitados""", postgres_connection(print_on=False))
                    greps_availables = np.array([state.DataClass.wh_list.index(wh) for wh in greps_availables.grep])

                progress_bar = st.progress(0)

                total_geus = sum(geu.cluster != 0 for geu in state.geus_by_id.values())

                env = state.env

                env_config = state.env_config
                env_config['is_streamlit'] = False
                env_config['bar_progress'] = None
                env_config['from_pickle'] = True

                trainer = set_neuronal_config(env_config)

                i=0
                for geu_id in state.geus_by_id.keys():
                #for i, geu_id in enumerate(['30']):

                    if state.geus_by_id[geu_id].cluster != 0:
                        i += 1
                        env.geu = env.data_class.geus_by_id[geu_id]
                        obs = env.reset()
                        # Action
                        if sum(env.stock_by_wh) > 0:
                            if state.geus_by_id[geu_id].cluster == 1:
                                if state.is_greps_available and state.cram_grep_option=='greps':
                                    action = env.action1_st(greps_availables)
                                else:
                                    action = env.action1()
                            else:
                                heuristic = trainer.compute_action(obs)
                                action = env.apply_action(heuristic)
                            state.initial_stock[geu_id] = np.append(np.copy(env.stock_by_wh), env.geu.domain)
                            state.distributions[geu_id] = translate_action(action, env)
                        else:
                            state.initial_stock[geu_id] = np.append(np.copy(env.stock_by_wh), env.geu.domain)
                            state.distributions[geu_id] = []

                        progress_bar.progress(i/total_geus)

                close_neuronal_connection()
                progress_bar.progress(100)

            # Initialize dataframe Output
            transit_stock_df = pd.DataFrame(index=state.env.data_class.wh_list, columns=state.env.data_class.wh_list)
            transit_stock_df = transit_stock_df.fillna(0)
            transit_stock_df.index.name = "Donating Wh"

            # Initial Stock DataFrame
            s0 = pd.DataFrame.from_dict(state.initial_stock, orient='index')
            s0.columns = state.env.data_class.wh_list + ['Dominio']
            for col in s0.columns[:-1]:
                s0[col] = s0[col].astype(int)

            # Filters-------------------------------------------------------------------------------------------------------
            # Multi Selector
            geus_ids = st.sidebar.multiselect('Seleccionar GEU, Nombre o Material',
                                              options=list(state.geus_by_id.keys()) + list(set(
                                                  list(state.geus_by_id.keys()) + [x.name for x in list(
                                                      state.geus_by_id.values())] + list(
                                                      state.DataClass.materials_by_id.keys())) - set(
                                                  list(state.geus_by_id.keys()))))

            # Domain
            domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
                       'TX - RADIOENLACES Y SATELITAL']
            domain = None
            if st.sidebar.checkbox('Filtrar por Dominio'):
                domain = st.sidebar.selectbox('Seleccionar Dominio', options=domains)

            # Cluster
            cluster = None
            if st.sidebar.checkbox('Filtrar por Cluster'):
                cluster = st.sidebar.selectbox('Seleccionar Cluster', options=['1 (baja)', '2 (media)', '3 (alta)'])

            # Criticality
            crit = None
            if st.sidebar.checkbox('Filtrar por Criticidad'):
                crit = st.sidebar.selectbox('Seleccionar Criticidad', options=['critico', 'mayor', 'bajo'])

            # procurement_types
            procurement_types = ['SPM', 'Buyable', 'Dismountable', 'Repairable']
            type = None
            if st.sidebar.checkbox('Filtrar por Abastecimiento'):
                type = st.sidebar.selectbox('Seleccionar Abastecimiento', options=procurement_types)

            # Amount
            amount = None
            if st.sidebar.checkbox('Filtrar por Cantidad'):
                amount = st.sidebar.slider('Cantidad Mínima',min_value=1, max_value=20,  step = 1)

            # Origin
            w0 = None
            if st.sidebar.checkbox('Filtrar por Grep Origen'):
                w0 = st.sidebar.selectbox('Seleccionar Grep', options=state.DataClass.wh_list, key='origin')

            # Destiyn
            wf = None
            if st.sidebar.checkbox('Filtrar por Grep Destino'):
                wf = st.sidebar.selectbox('Seleccionar Grep', options=state.DataClass.wh_list, key='destiny')

            # LeadTime
            lt = None
            if st.sidebar.checkbox('Filtrar por Leadtime'):
                lt = st.sidebar.slider('Cantidad Mínima',min_value=1, max_value=10,  step = 1)

            is_filtered = False
            if geus_ids or domain or cluster or crit or amount or type or w0 or wf or lt:
                is_filtered=True
                if geus_ids:
                    state.geu_filtered = geus_ids[0]
                    for multi in geus_ids:
                        if multi in state.DataClass.materials_by_id:
                            geus_ids[geus_ids.index(multi)] = state.DataClass.materials_by_id[multi].geu.id
                            continue
                        for it in [item.id for item in state.DataClass.geus_by_id.values() if multi == item.name]:
                            geus_ids[geus_ids.index(multi)] = it
                            break
                    geus_ids = list(set(geus_ids))
                    for multi in sorted(geus_ids, reverse=True):
                        if multi not in state.distributions:
                            geus_ids.remove(multi)
                    distributions = {geu_id: state.distributions[geu_id] for geu_id in geus_ids}
                    s0 = s0[s0.index.isin(geus_ids)]
                    result = generate_exportable_redistribution(distributions, state)

                elif domain or cluster or crit or amount or type or w0 or wf or lt:
                    distributions = state.distributions
                    result = generate_exportable_redistribution(distributions, state)
                    distributions = state.distributions
                    if domain:
                        result = result[result['Dominio'] == domain]
                        distributions = {geu_id: distributions[geu_id] for geu_id in distributions.keys()
                                         if state.DataClass.geus_by_id[geu_id].domain == domain}
                        s0 = s0[s0.index.isin([geu_id for geu_id in distributions.keys()
                                               if state.DataClass.geus_by_id[geu_id].domain == domain])]
                        is_filtered = True
                    if cluster:
                        result = result[result['Cluster'] == cluster]
                        if '1' in cluster:cluster = 1
                        elif '2' in cluster:cluster = 2
                        else: cluster = 3
                        distributions = {geu_id: distributions[geu_id] for geu_id in distributions.keys()
                                         if state.DataClass.geus_by_id[geu_id].cluster == int(cluster)}
                        s0 = s0[s0.index.isin([geu_id for geu_id in distributions.keys()
                                               if state.DataClass.geus_by_id[geu_id].cluster == int(cluster)])]
                        is_filtered = False
                    if type:
                        result = result[result['Tipo de Abastecimiento'] == type]
                        distributions = {geu_id: distributions[geu_id] for geu_id in distributions.keys()
                                         if state.DataClass.geus_by_id[geu_id].procurement_type == type}
                        s0 = s0[s0.index.isin([geu_id for geu_id in distributions.keys()
                                               if state.DataClass.geus_by_id[geu_id].procurement_type == type])]
                        is_filtered = False
                    if crit:
                        result = result[result['Criticidad'] == crit]
                        distributions = {geu_id: distributions[geu_id] for geu_id in distributions.keys()
                                         if state.DataClass.geus_by_id[geu_id].criticality == crit}
                        s0 = s0[s0.index.isin([geu_id for geu_id in distributions.keys()
                                               if state.DataClass.geus_by_id[geu_id].criticality == crit])]
                        is_filtered = False
                    if w0:
                        result = result[result['Origen'] == w0]
                        distributions = {geu_id: distributions[geu_id] for geu_id, list_of_transit in distributions.items()
                                         if [transit.amount for transit in list_of_transit if transit.donating_wh == w0]}
                        s0 = s0[s0.index.isin([geu_id for geu_id, list_of_transit in distributions.items()
                                               if [transit.amount for transit in list_of_transit
                                                   if transit.donating_wh == w0]])]
                        is_filtered = False
                    if wf:
                        result = result[result['Destino'] == wf]
                        distributions = {geu_id: distributions[geu_id] for geu_id, list_of_transit in distributions.items()
                                         if [transit.amount for transit in list_of_transit if transit.receiving_wh == wf]}
                        s0 = s0[s0.index.isin([geu_id for geu_id, list_of_transit in distributions.items()
                                               if [transit.receiving_wh for transit in list_of_transit
                                                   if transit.receiving_wh == wf]])]
                        is_filtered = False
                    if amount:
                        result = result[result['Cantidad'] >= amount]
                        distributions = {geu_id: distributions[geu_id] for geu_id, list_of_transit in distributions.items()
                                         if sum([transit.amount for transit in list_of_transit]) >= amount}
                        s0 = s0[s0.index.isin([geu_id for geu_id, list_of_transit in distributions.items()
                                               if sum([transit.amount for transit in list_of_transit]) >= amount])]
                        is_filtered = False
                    if lt:
                        result = result[result['Leadtime Estimado'] >= lt]
                        distributions = {geu_id: distributions[geu_id] for geu_id, list_of_transit in distributions.items()
                                         if [transit.amount for transit in list_of_transit
                                             if transit.leadtime >= lt]}
                        s0 = s0[s0.index.isin([geu_id for geu_id, list_of_transit in distributions.items()
                                               if [transit.amount for transit in list_of_transit
                                                   if transit.leadtime >= lt]])]
                        is_filtered = False
            else:
                distributions = state.distributions
                result = generate_exportable_redistribution(distributions, state)
            # End Filters---------------------------------------------------------------------------------------------------

            sf = s0.copy()

            # Fill values in DataFrame output
            if distributions or is_filtered:
                if is_filtered and not distributions:
                    st.warning("El material ingresado pertenece al cluster 0, por lo tanto, no se requiere redistribución.")
                    state.geus_by_id[state.geu_filtered].get_info(is_streamlit=True)

                    # Get Stock
                    state.env.geu = state.env.data_class.geus_by_id[state.geu_filtered]
                    obs = state.env.reset()
                    initial_stock={}
                    initial_stock[state.geu_filtered] = np.append(np.copy(state.env.stock_by_wh), state.env.geu.domain)

                    # Initial Stock DataFrame
                    s0 = pd.DataFrame.from_dict(initial_stock, orient='index')
                    s0.columns = state.env.data_class.wh_list + ['Dominio']
                    for col in s0.columns[:-1]:
                        s0[col] = s0[col].astype(int)

                    sf = s0.copy()

                    if s0.drop(['Dominio'], axis=1).sum().sum()==0:
                        st.info("El material no posee stock.")
                    else:
                        st.markdown('**Variación de Stock por Grep**')
                        st.plotly_chart(barchart_redistribution(s0, sf, is_filtered), use_container_width=True)

                else:

                    for geu_id, list_of_transit in distributions.items():
                        for transit_stock in list_of_transit:
                            transit_stock_df.at[transit_stock.donating_wh,
                                                transit_stock.receiving_wh] += transit_stock.amount

                            sf.at[geu_id, transit_stock.donating_wh] -= transit_stock.amount
                            sf.at[geu_id, transit_stock.receiving_wh] += transit_stock.amount

                    st.markdown('## Resumen')
                    mov = transit_stock_df.sum(axis=0).sum()
                    if mov == 1 :st.markdown(f'**Cantidad a Redistribuir: **{mov} unidad')
                    if mov != 1: st.markdown(f'**Cantidad a Redistribuir: **{mov} unidades')

                    total_geus = len([1 for value in distributions.values() if len(value)>0])
                    st.markdown(f'**Total de GEUs:** {total_geus}')

                    st.markdown(f'**Stock a la fecha:** {state.date}')

                    st.markdown('*Tabla de redistribuciones totalizada a todos los GEUs. En la primera columna se encuentra el grep donador. '
                                'En la primera fila se muestra el grep a recibir. '
                                'Los valores de la tabla indican la cantidad total a movilizar entre dichos almacenes (total de GEUs)*')

                    st.dataframe(transit_stock_df.style.background_gradient(cmap='Blues'))

                    if not is_filtered:
                        st.markdown('**Variación de Stock por Dominio y Grep**')
                        st.plotly_chart(barchart_redistribution(s0, sf, is_filtered), use_container_width=True)
                    else:
                        if s0.drop(['Dominio'], axis=1).sum().sum() == 0:
                            st.info("El material no posee stock.")
                        else:
                            st.markdown('**Variación de Stock por Grep**')
                            st.plotly_chart(barchart_redistribution(s0, sf, is_filtered), use_container_width=True)

                    #ray.shutdown()

                    st.markdown('## Redistribuciones Propuestas')
                    st.markdown('###### *Se muestran los primeros 10 registros de la tabla de resultados. '
                                'Para verla completa descargar el archivo* ')

                    st.table(result.head(10).style.hide_index())
                    show_dowload_option(generate_exportable_redistribution(state.distributions, state),
                                        f'redistribuciones_{state.date}')

                    st.markdown('## Visualización en Mapa')
                    st.plotly_chart(plot_redistribution_map(transit_stock_df,
                                                            state.env.grep_type, is_filtered,
                                                            state.format_map), use_container_width=True)
                    #if st.checkbox('Cambiar Formato del Mapa'): state.format_map = True
                    #else: state.format_map = False
                    st.markdown('## Redistribución por GREP')
                    st.plotly_chart(plot_redistribution_bar_plot(transit_stock_df), use_container_width=True)


def dashboard(state):
    style_title('Dashboard')
    st.markdown('Sección que genera los csv resultados de la simulación. Los csv deberán ser cargados en el Power BI. '
                'Se deben ingresar el stock inicial, las compras por año y los movimientos realizados.')

    st.header("Stock Inicial")
    list_of_stock = load_stock_dates()
    if len(list_of_stock)==1:
        date_of_stock = st.selectbox('Fecha de Stock', options=list_of_stock, index=0)
    else:
        date_of_stock = st.selectbox('Fecha de Stock', options=sorted(list_of_stock, reverse=True)[:5], index=1)
    start_date = dt.datetime(date_of_stock.year, date_of_stock.month, date_of_stock.day)

    st.header("Movimientos")
    movements = st.file_uploader(f"Ingresar movimientos", type=("xlsx"), key='2')

    resupplies_per_year = st.selectbox('Compras por año', options=[1, 2, 4])

    st.markdown('**Iniciar Simulación**')
    if st.button('Generar Exportables'):

        progress_bar = st.progress(0.1)
        initial_stock = create_query(f'select * from produccion.stock_s4 '
                                     f'where fecha = make_date({date_of_stock.year}, {date_of_stock.month}, {date_of_stock.day})',
                                     postgres_connection(is_streamlit=False))
        try:
            movements.seek(0)
        except:
            st.error("El CSV de Movimientos no fue ingresado.")
        today = dt.datetime(dt.datetime.today().year, dt.datetime.today().month, dt.datetime.today().day)

        progress_bar.progress(0.2)

        if not state.DataClass:
            state.DataClass = DataClass(start_date=start_date, is_streamlit=False, print_on=False, grep_type='greps', is_s4=True)
            #state.DataClass = DataClass.load_from_file(grep_type='greps')

        progress_bar.progress(0.3)

        history_teco = HistoryTeco(
            starting_date=dt.datetime(date_of_stock.year, date_of_stock.month, date_of_stock.day),
            end_date=today,
            ticket_sla=2,
            data_class=state.DataClass,
            df_movements=pd.read_excel(movements),
            initial_stock=initial_stock)

        history_teco.run()
        progress_bar.progress(0.45)
        history_dfs = history_teco.generate_exportables_dataframes()
        progress_bar.progress(0.55)

        env_config = {"start_date": start_date, "end_date": today, "get_info": True,
                      'grep_type': 'greps', "from_pickle": False, 'use_historic_stocks': True,
                      'resupplies_per_year': resupplies_per_year, "is_s4": True}

        model_vs_history = ModelVSHistory(env_config, chosen_heuristics=['Action1'])
        model_vs_history.run(clusters=[1])
        df_service_1 = model_vs_history.service_data.copy()
        df_stock_1 = model_vs_history.stock_data.copy()

        progress_bar.progress(0.7)

        model_vs_history = ModelVSHistory(env_config, chosen_heuristics=['Neural Net'])
        model_vs_history.run(clusters=[2, 3])
        ray.shutdown()
        model_vs_history.add_history_dash(history_dfs, with_cluster_0=True)

        progress_bar.progress(0.85)

        service_data = model_vs_history.service_data.append(df_service_1)
        stock_data = model_vs_history.stock_data.append(df_stock_1)

        service_data.loc[service_data['heuristica'] != 'historia', 'heuristica'] = 'NeuralTeco'
        stock_data.loc[stock_data['heuristica'] != 'historia', 'heuristica'] = 'NeuralTeco'

        dates = service_data.loc[service_data['heuristica'] == 'historia', 'fecha'].unique()
        stock_data = stock_data[stock_data['fecha'].isin(dates)]

        # service_data[['on_time', "total"]] = service_data.groupby(['heuristica', "cram", "geu"])[['on_time', "total"]].transform(
        #     pd.Series.cumsum)

        geus_info = generate_pbi_geus_info(state.DataClass.geus_by_id)

        progress_bar.progress(0.98)

        tables = (('stock', stock_data, ['String']*2 + ['Date'] + ['String']*3),
                  ('service', service_data, ['String']*2 + ['Date'] + ['String']*4),
                  ('pbi_geus_info', geus_info, ['String']*12))

        for name, df, types in tables:
            my_path = os.path.join(os.path.dirname(__file__), 'dashboard_csv/')
            df.to_csv(f'{my_path}{name}.csv')

        STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
        DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
        if not DOWNLOADS_PATH.is_dir():
            DOWNLOADS_PATH.mkdir()

        if os.path.exists(str(DOWNLOADS_PATH / f"dashboard_csv.zip")):
            os.remove(str(DOWNLOADS_PATH / f"dashboard_csv.zip"))

        # create a ZipFile object
        with ZipFile(str(DOWNLOADS_PATH / f"dashboard_csv.zip"), 'w') as zipObj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(os.path.join(os.path.dirname(__file__), 'dashboard_csv/')):
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    # Add file to zip
                    zipObj.write(filePath, basename(filePath))

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
        st.markdown(
            f'<a href="downloads/dashboard_csv.zip" download="dashboard_csv.zip" style="{style}">Descargar Dashboards_csv</a>',
            unsafe_allow_html=True)

        progress_bar.progress(1.0)
        st.success('Proceso terminado, ingresar al Power BI y actualizar los datos para ver el dashboard')


# ----------------------------------------------------------------------------------------------------------------------

def clean_state(state):
    state.geus_by_id = {}
    state.DataClass = None
    state.test = 0
    state.is_first_state = False
    state.is_first_geu_info = True
    state.figures = {}
    state.env = None
    state.distributions = {}
    state.initial_stock = {}
    state.purchase = None
    state.has_purchase = False
    state.button_save = False
    set_multiplier_to_deafult(type='low', resupplies_per_year=1, grep_type='greps')
    state.multipliers = None
    state.cluster = None
    state.button_save = None
    state.ss = None
    state.date = None
    state.env_config = None
    state.format_map = False
    state.buy = None
    state.is_greps_available = None

def show_cleaning_button(state):
    clean = st.sidebar.button('Limpiar Memoria')
    if clean:
        clean_state(state)


if __name__ == "__main__":
    st_main()