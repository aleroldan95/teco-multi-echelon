import pickle as pkl

import networkx as nx
import numpy as np
import streamlit as st

import functions
from classes.material import Material
from classes.ticket import Ticket
from clustering import *


class DataClass:

    #  ============================================== Static Attributes ==============================================

    procurement_types = ['SPM', 'Buyable', 'Dismountable', 'Repairable']
    # domains = ['ACCESO - FIJA', 'ACCESO - MOVIL', 'CORE VOZ', 'ENTORNO', 'TRANSPORTE - DX', 'TRANSPORTE - TX',
    #     #            'TX - RADIOENLACES Y SATELITAL']
    clusters = [1, 2, 3]

    #  ============================================ Load from pickle file ============================================
    @staticmethod
    def load_from_file(filename: str = 'last_data_model', grep_type: [str] = '7_crams'):
        try:
            if grep_type == '7_crams':
                filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '_cram.pkl')
            elif grep_type == 'greps':
                filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '.pkl')
            elif grep_type == 'unicram':
                filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '_unicram.pkl')
            with open(filedir, 'rb') as file:
                model = pkl.load(file)
            print('Loaded data from pickle  [Done]')
            return model
        except FileNotFoundError:
            raise FileNotFoundError("No se encontro archivo pickle: {}".format(filename))

    #  ================================================= Constructor =================================================

    def __init__(self, print_on: bool = True, is_streamlit: bool = False, bar_progress: st = None,
                 grep_type: str = '7_crams', demand_type: str = 'tickets', start_date = dt.datetime(2018,7,1),
                 is_s4: bool = False):

        """
        :param print_on: print on console
        :param is_streamlit: print on streamlit
        :param bar_progress: streamlit parameter
        :param grep_type: ['unicram', '7_crams', 'greps']
        :param demand_type: 'tickets' as demand; 'mov_351' as demand
        """

        # Booleans
        self.print_on = print_on
        self.is_streamlit = is_streamlit
        self.grep_type = grep_type
        self.demand_type = demand_type
        self.start_date = start_date
        self.is_s4 = is_s4

        # Database Connection
        self.engine = functions.postgres_connection(self.is_streamlit, self.print_on)

        # Warehouse List
        self.relation_grep_wh, self.wh_list = self.load_relation_grep_warehouses()
        """self.wh_list: [str] = ['CAPITAL', 'RESI', 'COR', 'CUYO', 'GBA', 'JUJUY-SALTA', 'MDQ', 'MISIONES', 'NEUQUEN',
                                   'CORRIENTES', 'San Pedro', 'ROS', 'SANTA FE', 'TUC', '9 de Julio', 'Bahia Blanca',
                                   'VENADO TUERTO', 'CONCORDIA']"""

        # Data Dictionaries
        self.materials_by_id: {str: Material}
        self.geus_by_id: {str: GEU}
        self.leadtime_between_warehouses: {(str, str): int}
        self.df_leadtimes: pd = pd.DataFrame()
        self.geus_with_multiple_domains = []

        # Metadata
        self.k_clusters: int

        # Progress bar
        total_processes_range = range(0, 16)
        progress_step = int(100 / len(total_processes_range))
        accum = self.update_progress_bar(bar_progress, progress_step)

        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        if self.print_on: print("\n 1. Material Processing\n")             # 1. MATERIALS
        if self.is_streamlit: st.write("**1. Material Processing**")

        df_materials = self._load_materials()                              # ---1. Load material info
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self._instantiate_materials(df_materials)                          # ---2. Instantiate Material Objects
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self._set_missing_prices(df_materials)                             # ---3. Set Material missing prices
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        df_leadtimes, df_lt_spm, df_lt_repair = self._load_leadtimes()     # ---4. Set Leadtimes
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self._process_leadtimes(df_leadtimes, df_lt_spm, df_lt_repair)
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        if self.demand_type == 'mov_351':
            df_demand = self._load_demand()                                    # ---5. Set Demand
            accum = self.update_progress_bar(bar_progress, accum, progress_step)
        else:
            df_demand = None
        df_tickets = self._load_tickets()
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self._process_demand(df_demand, df_tickets)
        accum = self.update_progress_bar(bar_progress, accum, progress_step)

        # --------------------------------------------------------------------------------------------------------------
        message = "Process Materials Finished"
        functions.elapsed_time_message(False, message=message, tic=tic, is_end=True,
                                       print_on=self.print_on, is_streamlit=self.is_streamlit)
        #                                       =====================================

        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        if self.print_on: print("\n 2. GEUs Processing\n")               # 2. GEUs
        if self.is_streamlit: st.write("**2. GEUS Processing**")

        self.domain_list = self._load_domains()                          # ---1. Load accepted domains
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        df_ges = self.load_ges()                                         # ---2. Load GEs
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        df_geus = self.calculate_geus(df_ges)                            # ---3. Calculate GEUs
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self.instantiate_geus(df_geus)                                   # ---4. Instantiate GEU Objects
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self.create_unigeus()                                            # ---5. Instantiate Single Mat GEUs
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self.complete_geu_information(df_lt_spm, df_lt_repair)                        # ---6. Complete GEU Object info
        accum = self.update_progress_bar(bar_progress, accum, progress_step)

        self.df_initial_stocks = self._set_initial_stocks()

        # --------------------------------------------------------------------------------------------------------------
        message = "Process GEUs Finished"
        functions.elapsed_time_message(False, message=message, tic=tic, is_end=True,
                                       print_on=self.print_on, is_streamlit=self.is_streamlit)

        #                                       =====================================

        tic = functions.elapsed_time_message(True)  # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        if self.print_on: print("\n 3. Demand Processing\n")            # 3. Demand: Wh, tickets, clusters
        if self.is_streamlit: st.write("**3. Demand Processing**")

        self.process_warehouses_leadtimes()                              # ---6. Leadtimes
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self.process_tickets(df_tickets)                                 # ---7. Load Tickets
        accum = self.update_progress_bar(bar_progress, accum, progress_step)
        self.set_geu_clusters()                                          # ---8. K-means-clustering
        self.update_progress_bar(bar_progress, accum, progress_step, is_end=True)
        # --------------------------------------------------------------------------------------------------------------
        message = "Demand Processing Finished"
        functions.elapsed_time_message(False, message=message, tic=tic, is_end=True,
                                       print_on=self.print_on, is_streamlit=self.is_streamlit)

        self.reorder_wh_list()

        # Met_geu_to_sql
        self.mat_geu_to_sql()

        if self.is_streamlit:
            st.success('Carga de Datos Finalizada')
            st.balloons()
        # ================================================ Save Data =================================================

        self.save()

    #  ============================================== Material Processing ============================================

    def _load_materials(self):
        """
        Generate a dataframe with materials' information
        :return: material DF
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Query material table in db
        query = """SELECT DISTINCT * FROM produccion.materiales;"""
        df_materials = functions.create_query(query, self.engine)
        df_materials['Dominio'] = df_materials['Dominio'].apply(lambda x: x.upper())

        # Check s4
        if self.is_s4:
            catalog = df_materials['Catalogo']
            df_materials['Catalogo'] = df_materials['Catalogo_S4']
            df_materials['Catalogo_S4'] = catalog

        # Check empty catalog material
        if (df_materials['Catalogo'].isnull().sum() > 0) or (
                df_materials['Catalogo'].isna().sum() > 0) or ((df_materials['Catalogo'] == '').sum() > 0):
            st.error('Se encontraron Materiales sin Catálogos (registros vacíos). Ir a la sección "Cargar nuevos Datos" '
                     'y completar los catálogos nulos de la tabla "Materiales". Se omitieron dichos registros.')


        # Check for duplicated materials
        mat_value_counts = df_materials['Catalogo'].value_counts()
        duplicated_mats = []
        for i, v in mat_value_counts.iteritems():
            if v > 1:
                duplicated_mats.append(i)
            else:
                break

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Materials..."                                                                    # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic, is_streamlit=self.is_streamlit)
        if duplicated_mats:
            if self.is_streamlit:
                st.warning(f'Catálogo de Material Duplicado: {duplicated_mats}')
            print('Duplicated Catalogs in info:', duplicated_mats)

        return df_materials

    def _instantiate_materials(self, df_materials: pd.DataFrame):
        """
        Create Material Class and return a dictionary with all material objects
        :return: Dictionary {id. , mat}
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        self.materials_by_id = {}
        for index, row in df_materials.iterrows():
            if row['Catalogo'] != '' or row['Catalogo'] != None:
                self.materials_by_id.update({str(row["Catalogo"]): Material.create_material_from_df(row)})


        mats_without_criticality = sum(1 for material in self.materials_by_id.values() if not material.has_criticality_data)
        if mats_without_criticality > 1 and self.is_streamlit:
            st.warning(f"Se encontraron {mats_without_criticality} material/es sin criticidad asignada. "
                       f"La Criticidad puede ser ingresada de la siguiente manera: "
                       f"'critico', 'critica', 'alta', 'alto', 'mayor', 'media', 'medio', 'bajo', 'baja'. "
                       f"Se le asigna criticidad Baja")

        # Materials without procurement type
        mats_without_ptype = [mat.catalog for mat in self.materials_by_id.values() if not mat.has_procurement_type]

        # --------------------------------------------------------------------------------------------------------------
        message = "Instantiating Material objects..."                                                       # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)
        if mats_without_ptype:
            if self.is_streamlit:
                st.warning(f"Se encontraron {len(mats_without_ptype)} material/es sin Tipo de Abastecimiento correcto:"
                           f"{mats_without_ptype}"
                           f"Tipo de Abastecimiento debe contener al menos 1 de las siguientes palabras:"
                           f"[SPM, RMA, COMPRABLE, REPARABLE, DESMONTABLE]")
            else:
                print(f"Se encontraron {mats_without_ptype} material/es sin Tipo de Abastecimiento correcto."
                      f"{mats_without_ptype}")

    def _set_missing_prices(self, df_materials: pd.DataFrame):
        """
        Sets price for materials with null values.
        If there are other materials with the same model, the models mean price is assigned.
        Else, the domain mean price is assigned.
        :return: Void
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------
        #Price

        # Trim DataFrame
        prices_df = df_materials[['Catalogo', 'Precio', 'Unidad_Precio', 'Equipo / Modelo', 'Dominio', 'TC']]
        prices_df_full = prices_df[prices_df["Precio"] > 0]
        for index, row in prices_df_full.iterrows():
            if row['Unidad_Precio'] in ['ars', 'ARS']:
                row['Precio'] = row['Precio'] / row['TC']

        # Calculate mean prices by model (there may be many actual models inside an equipment string description)
        models_with_prices = {}
        for mat in self.materials_by_id.values():
            if mat.price is not None:
                for equip in mat.equipments:
                    # Add equipment if needed
                    if equip not in models_with_prices:
                        models_with_prices[equip] = [0, 0]

                    # Update value
                    models_with_prices[equip][0] += mat.price
                    models_with_prices[equip][1] += 1
        # Take the average price
        for model in models_with_prices.values():
            model[0] /= model[1]

        # Calculate mean prices by domain
        df_prices_gb_domain = prices_df_full[['Dominio', 'Precio']].groupby('Dominio').mean()
        df_prices_gb_domain.reset_index(inplace=True)

        # Calculate mean price
        mean_price = prices_df_full['Precio'].mean()
        # print('Mean price:', mean_price)

        # Set inferred price (if needed)
        for mat in self.materials_by_id.values():
            if mat.price is None:
                # Model price
                matching_equipments = [i for i in mat.equipments if i in models_with_prices.keys()]
                if matching_equipments:
                    # Take average known model price
                    model_price, datapoints = 0, 0
                    for equip in matching_equipments:
                        model_price += models_with_prices[equip][0]
                        datapoints += models_with_prices[equip][1]
                    model_price /= datapoints

                    # Set model price
                    mat.set_price(model_price, price_type='2-Model')

            if mat.price is None:
                # Domain Price
                try:
                    domain_price = float(
                        df_prices_gb_domain[df_prices_gb_domain['Dominio'] == mat.domain].Precio)
                    mat.set_price(domain_price, price_type='1-Domain')
                except:
                    pass
            if mat.price is None:
                # Mean Price
                mat.set_price(mean_price, price_type='0-Mean')

        #---------------------------------------------------------------------------------------------------------------
        #Rep

        # Trim DataFrame
        prices_df = df_materials[['Catalogo', 'Costo_Reparaciones', 'Unidad_Reparacion', 'Equipo / Modelo', 'Dominio']]
        prices_df_full = prices_df[prices_df["Costo_Reparaciones"] > 0]

        for index, row in prices_df_full.iterrows():
            if row['Unidad_Reparacion'] in ['usd', 'USD']:
                row['Costo_Reparaciones'] = row['Costo_Reparaciones'] * row['TC']

        # Calculate mean prices by model (there may be many actual models inside an equipment string description)
        models_with_prices = {}
        for mat in self.materials_by_id.values():
            if mat.rep_price is not None:
                for equip in mat.equipments:
                    # Add equipment if needed
                    if equip not in models_with_prices:
                        models_with_prices[equip] = [0, 0]

                    # Update value
                    models_with_prices[equip][0] += mat.rep_price
                    models_with_prices[equip][1] += 1
        # Take the average price
        for model in models_with_prices.values():
            model[0] /= model[1]

        # Calculate mean prices by domain
        df_prices_gb_domain = prices_df_full[['Dominio', 'Costo_Reparaciones']].groupby('Dominio').mean()
        df_prices_gb_domain.reset_index(inplace=True)

        # Calculate mean price
        mean_price = prices_df_full['Costo_Reparaciones'].mean()
        # print('Mean price:', mean_price)

        # Set inferred price (if needed)
        for mat in self.materials_by_id.values():
            if mat.rep_price is None:
                # Model price
                matching_equipments = [i for i in mat.equipments if i in models_with_prices.keys()]
                if matching_equipments:
                    # Take average known model price
                    model_price, datapoints = 0, 0
                    for equip in matching_equipments:
                        model_price += models_with_prices[equip][0]
                        datapoints += models_with_prices[equip][1]
                    model_price /= datapoints

                    # Set model price
                    mat.set_rep_price(model_price, price_type='2-Model')

            if mat.rep_price is None:
                # Domain Price
                try:
                    domain_price = float(
                        df_prices_gb_domain[df_prices_gb_domain['Dominio'] == mat.domain].Costo_Reparaciones)
                    mat.set_rep_price(domain_price, price_type='1-Domain')
                except:
                    pass
            if mat.rep_price is None:
                # Mean Price
                mat.set_rep_price(mean_price, price_type='0-Mean')

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Prices..."                                                                       # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    def _load_leadtimes(self):
        """
        Generate a dataframe with materials' information
        :return:
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        query = """SELECT * FROM produccion.oes_s4;"""
        df_leadtimes = functions.create_query(query, self.engine)
        if self.is_s4:
            df_leadtimes['Material'] = df_leadtimes['Catalogo_S4']

        query = """SELECT * FROM produccion.leadtimes_spm;"""
        df_leadtimes_spm = functions.create_query(query, self.engine)
        df_leadtimes_spm.set_index(['dominio', 'marca'], drop=True, inplace=True)

        query = """SELECT * FROM produccion.leadtimes_reparables;"""
        df_leadtimes_repairables = functions.create_query(query, self.engine)
        df_leadtimes_repairables.set_index('dominio', drop=True, inplace=True)

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Lead Times..."                                                                   # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

        return df_leadtimes, df_leadtimes_spm, df_leadtimes_repairables

    def _process_leadtimes(self, df_leadtimes: pd.DataFrame, df_lt_spm: pd.DataFrame, df_lt_repair: pd.DataFrame):
        """
        Set the lead time and lead time deviation for every material
        :return: Set lead times
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Downsize df to the columns that are needed
        # TODO: Why are we using a DataFrame in from_dict method?
        df_aux = pd.DataFrame.from_dict(df_leadtimes, orient='columns', dtype=None, columns=None)
        df_aux = df_aux[['Documento', 'Proveedor', 'Fecha_entrega',
                         'Fecha_doc', 'Material', 'Descripcion']]
        df_aux['Material'] = df_aux['Material'].astype(str)

        # Find materials' brands
        aux_dict = {m_id: material.brand for m_id, material in self.materials_by_id.items()}
        df_materials_by_id = pd.DataFrame(list(aux_dict.items()))
        df_materials_by_id.columns = ["ID", "Marca"]

        # Add brands to df
        df_aux = pd.merge(left=df_aux, right=df_materials_by_id, left_on='Material', right_on='ID', how='left')

        # Calculate LTs
        df_aux['LT'] = df_aux['Fecha_entrega'] - df_aux['Fecha_doc']

        # Average LT by brand
        df_aux_brands = df_aux[['Marca', 'LT']].copy()
        df_aux_brands['LT'] = df_aux_brands['LT'].dt.days.astype('int16')
        df_brands = df_aux_brands.groupby(df_aux_brands['Marca'])['LT'].agg(["mean", "std"]).copy()
        df_brands.columns = ["Promedio_marca", "Desvio_marca"]

        # Average LT by supplier
        df_aux_suppliers = df_aux[['Proveedor', 'LT']].copy()
        df_aux_suppliers['LT'] = df_aux_suppliers['LT'].dt.days.astype('int16')
        df_suppliers = df_aux_suppliers.groupby(
            df_aux_suppliers['Proveedor'])['LT'].agg(["mean", "std"]).copy()
        df_suppliers.columns = ["Promedio_proveedor", "Desvio_proveedor"]

        # Average LT by material
        df_aux_materials = df_aux[['Material', 'LT']].copy()
        df_aux_materials['LT'] = df_aux_materials['LT'].dt.days.astype('int16')
        df_materials = df_aux_materials.groupby(df_aux_materials['Material'])['LT'].agg(['mean', 'std']).copy()
        df_materials.columns = ["Promedio_material", "Desvio_material"]

        # Add columns to df_materials: supplier, sd of supplier
        df_materials = pd.merge(left=df_materials, right=df_aux, left_on='Material', right_on='Material')
        df_materials = pd.merge(left=df_materials, right=df_suppliers, left_on='Proveedor', right_on='Proveedor')
        # Adding column: Replacing NaN-filled sd with the supplier's sd
        df_materials.loc[df_materials["Desvio_material"].isna(), "Desvio"] = df_materials.loc[
            df_materials["Desvio_material"].isna(), "Desvio_proveedor"]
        df_materials.loc[df_materials["Desvio_material"].notna(), "Desvio"] = df_materials.loc[
            df_materials["Desvio_material"].notna(), "Desvio_material"]

        # Keep relevant columns
        df_materials = df_materials[["Material", "LT", "Desvio"]]
        df_materials['LT'] = df_materials['LT'].dt.days.astype('int16')

        # Add materials w/o LT
        df_materials_by_id2 = pd.merge(left=df_materials_by_id, right=df_materials, left_on='ID', right_on='Material',
                                       how='left')
        # Add sd by brand
        df_materials_by_id2 = pd.merge(left=df_materials_by_id2, right=df_brands, left_on='Marca', right_on='Marca',
                                       how='left')
        # Add lt and sd by brand
        df_materials_by_id2.loc[df_materials_by_id2["LT"].isna(), "LT2"] = df_materials_by_id2.loc[
            df_materials_by_id2["LT"].isna(), "Promedio_marca"]
        df_materials_by_id2.loc[df_materials_by_id2["LT"].notna(), "LT2"] = df_materials_by_id2.loc[
            df_materials_by_id2["LT"].notna(), "LT"]
        df_materials_by_id2.loc[df_materials_by_id2["Desvio"].isna(), "Desvio2"] = df_materials_by_id2.loc[
            df_materials_by_id2["Desvio"].isna(), "Desvio_marca"]
        df_materials_by_id2.loc[df_materials_by_id2["Desvio"].notna(), "Desvio2"] = df_materials_by_id2.loc[
            df_materials_by_id2["Desvio"].notna(), "Desvio"]

        # Calculate general average and sd
        df_medias = df_aux_materials['LT'].agg({"Promedio_general": "mean", "Desvio_general": "std"}).copy()

        # Add general average and sd as columns
        df_materials_by_id2['Promedio_general'] = df_medias["Promedio_general"]
        df_materials_by_id2['Desvio_general'] = df_medias["Desvio_general"]
        df_materials_by_id2.loc[df_materials_by_id2["Desvio2"].isna(), "Desvio2"] = df_materials_by_id2.loc[
            df_materials_by_id2["Desvio2"].isna(), "Desvio_general"]
        df_materials_by_id2.loc[df_materials_by_id2["LT2"].isna(), "LT2"] = df_materials_by_id2.loc[
            df_materials_by_id2["LT2"].isna(), "Promedio_general"]

        # Final data cleaning
        df_materials_final = df_materials_by_id2[['ID', 'LT2', 'Desvio2']].copy()
        df_materials_final.columns = ['Material', 'LT', 'Desvio_LT']
        df_materials_final['LT'] = df_materials_final['LT'].apply(np.ceil)
        df_materials_final['Desvio_LT'] = df_materials_final['Desvio_LT'].apply(np.ceil)

        # Set leadtimes
        for (material_id, leadtime_mean, leadtime_deviation) \
                in zip(df_materials_final["Material"], df_materials_final["LT"], df_materials_final["Desvio_LT"]):
            material = self.materials_by_id[material_id]
            material.set_leadtimes(leadtime_mean, leadtime_deviation)

        # --------------------------------------------------------------------------------------------------------------
        message = "Processing Lead Times..."                                                                # Stop Timer
        functions.elapsed_time_message(False, message, tic, print_on=self.print_on, is_streamlit=self.is_streamlit)

    def _load_demand(self):
        """
        Generate a dataframe with materials' demand
        :return: pd.DataFrame with demand history
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------
        # Create query
        if self.grep_type in ['7_crams', 'unicram']:
            query = """SELECT * FROM p03_daily_demand_cram;"""
        else:
            query = """SELECT * FROM p03_daily_demand_grep;"""

        df_demand = functions.create_query(query, self.engine)
        df_demand["fecha"] = pd.to_datetime(df_demand["fecha"])

        if self.grep_type == 'unicram': df_demand['grep'] == '00'

        # Use only materials in scope
        list_of_materials = list(self.materials_by_id.keys())
        df_demand = df_demand[df_demand.material.isin(list_of_materials)]

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Demand Data..."                                                                  # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

        return df_demand

    def _load_tickets(self):
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Load tickets
        #if self.grep_type in ['7_crams', 'unicram']:
        #    query = f"""SELECT * FROM p02_tickets_by_cram;"""
        #else:
        if self.is_s4:
            query = f"""SELECT * FROM produccion.tickets_s4;"""
        else:
            query = f"""SELECT * FROM produccion.tickets;"""
        df_tickets = functions.create_query(query, self.engine)
        df_tickets["fecha"] = pd.to_datetime(df_tickets["fecha"])
        df_tickets.columns = ['fecha', 'material', 'cantidad', 'grep']

        greps_banned = [wh for wh in list(df_tickets['grep'].unique()) if wh not in self.relation_grep_wh.keys()]
        if self.is_streamlit and greps_banned:
            st.warning(f"Las siguientes relaciones de centro-almacén no se encontraron en la base de datos relacion_grep_almacen: {greps_banned}. "
                       f"No serán tenidos en cuenta. Para que sean incluidos se deben agregar dichas relaciones con su grep correspondiente en relacion_grep_alamcen")
        df_tickets = df_tickets[~df_tickets['grep'].isin(greps_banned)]

        df_tickets['grep'] = df_tickets['grep'].map(lambda x: self.relation_grep_wh[x])

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Tickets Data..."  # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

        return df_tickets

    def _process_demand(self, df_demand, df_tickets):
        """
        Adding Material's demand
        :param df_demand:
        :return:
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        if self.demand_type == 'mov_351':
            for date, material, warehouse, demand in df_demand.values:
                self.materials_by_id[material].add_demand(functions.to_datetime(date), warehouse, demand)

        elif self.demand_type == 'tickets':
            for x, row in df_tickets.iterrows():
                if row['material'] in self.materials_by_id.keys():
                    self.materials_by_id[row['material']].add_demand(functions.to_datetime(row['fecha']),
                                                                     row['grep'], row['cantidad'])

        # --------------------------------------------------------------------------------------------------------------
        message = "Processing Demand..."                                                                    # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    #  ================================================ GEU Processing ===============================================
    def load_relation_grep_warehouses(self):
        query = """SELECT grep, ce_alm_s44, cram FROM produccion.relacion_grep_almacen;"""
        df = functions.create_query(query, self.engine)
        if self.grep_type == 'greps':
            return {row['ce_alm_s44']: row['grep'] for index, row in df.iterrows()},\
                   sorted(list(df['grep'].unique()))
        elif self.grep_type == '7_crams':
            return {row['ce_alm_s44']: row['cram'] for index, row in df.iterrows()},\
                   sorted(['00', '01', '02', '03', '04', '05', '06'])
        elif self.grep_type == 'unicram':
            return {row['ce_alm_s44']: '00' for index, row in df.iterrows()},\
                   ['00']
        raise NotImplemented('Grep type not found: ' + self.grep_type)

    def _load_domains(self):
        """
        Domains
        :return: material DF
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Query material table in db
        query = """SELECT "Dominio" FROM r00_domain;"""
        accepted_domains = functions.create_query(query, self.engine)
        accepted_domains = list(accepted_domains['Dominio'])
        accepted_domains = [x.upper() for x in accepted_domains]

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Materials..."                                                                    # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic, is_streamlit=self.is_streamlit)

        return accepted_domains

    def load_ges(self):
        """
        Generate a dataframe with equivalences' information from DB
        :return: pd.DataFrame
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        if self.is_s4:
            query = """SELECT "CatS4", "EquivS4", "GrupoEq" ge FROM produccion.equivalencias where "CatS4" is not null and "EquivS4" is not null and "GrupoEq" is not null;"""
        else:
            query = """SELECT "Catalogo", "Equivalente", "GrupoEq" ge FROM produccion.equivalencias;"""
        df_ges = functions.create_query(query, self.engine)

        if self.is_s4:
            df_ges.columns = ["Catalogo", "Equivalente", "ge"]

        # --------------------------------------------------------------------------------------------------------------
        message = "Loading Equivalences..."                                                                 # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

        return df_ges

    def calculate_geus(self, df_ges):
        """
        Automatically assigns GEUs using mainly the library NetworkX (nx)
        :param df_ges: pd.DataFrame
        :return: pd.DataFrame
        """

        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        df_ges = df_ges.astype({'Catalogo': 'str', 'Equivalente': 'str'})

        # Step 1: Filter materials within scope

        df_ges = df_ges[df_ges['Catalogo'].apply(lambda x: str(x) in self.materials_by_id.keys())]
        df_ges = df_ges[df_ges['Equivalente'].apply(lambda x: str(x) in self.materials_by_id.keys())]
        df_ges.reset_index(inplace=True)

        # Step 2: Calculate minimum ge for each material

        # Load both columns
        df_catalog = df_ges.loc[:, ['Catalogo', 'ge']]
        df_catalog.rename(columns={"Catalogo": "material"}, inplace=True)
        df_equivalent = df_ges.loc[:, ['Equivalente', 'ge']]
        df_equivalent.rename(columns={"Equivalente": "material"}, inplace=True)

        # Append the data of both columns
        df_material = df_catalog.append(df_equivalent)
        df_material.drop_duplicates(subset=['material', 'ge'], keep='first', inplace=True)

        # Find minimum ges
        df_material = df_material.groupby('material').min().sort_values('ge')
        df_material.rename(columns={"ge": "min_ge"}, inplace=True)

        # Step 3: Find GEUs (i.e. graph islands)
        main_graph = nx.Graph()
        for index, row in df_ges.iterrows():
            orig = str(row['Equivalente'])
            dest = str(row['Catalogo'])
            main_graph.add_edge(orig, dest)

        islands = [main_graph.subgraph(c) for c in nx.connected_components(main_graph)]

        # Step 4: Name GEUs using minimum GEs
        island_dict = {}
        for i in range(len(islands)):
            prev_size = len(island_dict)
            # Select appropriate GEU number
            geu = 999999
            for node in islands[i].nodes:
                if df_material.loc[node]['min_ge'] < geu:
                    geu = df_material.loc[node]['min_ge']

            # Beware of false GEUs: GEUs forced by errors in the data ('joining' 2 unrelated real GEUs)
            if str(geu) not in island_dict.keys():
                island_dict[str(geu)] = []
                for node in islands[i]:
                    island_dict[str(geu)].append(node)
            else:
                island_dict[str(geu) + '-B'] = []
                for node in islands[i]:
                    island_dict[str(geu) + '-B'].append(node)

            # To be extra sure, we include a print that is usually triggered by false GEUs
            if len(island_dict) == prev_size:
                if self.print_on: print('GEU', geu, 'has not changed islands_dic size')

        # This number can't change. If it did there's a problem.
        warning_1 = '\t\t\t\t\t\t\t\t\t   Islands:\t\t  ' + str(len(islands)) + "✓ [Checked]" \
            if len(islands) == len(island_dict) else "⚠ [Warning]"
        if self.print_on: print(warning_1)
        if self.is_streamlit: st.write(warning_1)
        warning_2 = '\t\t\t\t\t\t\t\t\t   Islands (GEUs):' + str(len(island_dict)) + "✓ [Checked]" \
            if len(islands) == len(island_dict) else "⚠ [Warning]"
        if self.print_on: print(warning_2)
        if self.is_streamlit: st.write(warning_2)

        # Step 5: Make GEUs graphs
        geus_in_graphs = {}
        for geu in island_dict.keys():
            # Initiate GEU's DiGraph
            geus_in_graphs[geu] = nx.DiGraph()

            for row in range(df_ges.shape[0]):
                orig = str(df_ges['Equivalente'][row])

                # We can assume that if orig is in the GEU, then the dest must also be in it
                if orig in island_dict[geu]:
                    dest = str(df_ges['Catalogo'][row])

                    geus_in_graphs[geu].add_edge(orig, dest)

        # Step 6: Extract material-geu relations
        df_automatic_geus = pd.DataFrame(columns=['material', 'geu'])
        row = 0
        for geu in island_dict.keys():
            for mat in island_dict[geu]:
                df_automatic_geus.loc[row] = [str(mat), geu]
                row += 1

        # Step 7: Make new df_geus
        df_new_alternatives = df_ges
        df_new_alternatives['geu'] = df_new_alternatives['Catalogo'].apply(lambda x:
                                                                           list(df_automatic_geus[
                                                                                    df_automatic_geus['material'] ==
                                                                                    str(x)]['geu'].values)[0])

        # --------------------------------------------------------------------------------------------------------------
        message = "Calculating GEUs..."                                                                     # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

        return df_new_alternatives

    def instantiate_geus(self, df_geus):
        """
        Instantiate GEU objects
        :param df_geus: dataframe with geus info
        :return: {id, GEU}
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        self.geus_by_id = {}
        for index, row in df_geus.iterrows():

            # Check if materials are in scope
            if (str(row['Catalogo']) in self.materials_by_id) and str(row['Equivalente']) in self.materials_by_id:

                # Check if geu object was created already
                mat_1 = self.materials_by_id[f'{row["Catalogo"]}']
                mat_2 = self.materials_by_id[f"{row['Equivalente']}"]

                if str(row['geu']) in self.geus_by_id:
                    # If already exists, the materials are added to the GEU (as it is a set,
                    # there are no repeated values)
                    self.geus_by_id[str(row['geu'])].add_material_equivalence(mat_1)
                    self.geus_by_id[str(row['geu'])].add_material_equivalence(mat_2)

                else:
                    # If it doesn't exist, a new GEU object is created
                    list_of_materials = list([mat_1, mat_2])
                    geu = GEU(group_id=row['geu'], list_of_materials=list_of_materials)
                    self.geus_by_id[str(geu.id)] = geu

                    # Set cardinality
                    self.geus_by_id[str(geu.id)].is_unigeu = len(self.geus_by_id[str(geu.id)].materials) == 1

        # Iterate geus
        for geu in self.geus_by_id.values():
            # Make GEUs graphs
            geu.graph = nx.DiGraph()

            for row in range(df_geus.shape[0]):
                orig = str(df_geus['Equivalente'][row])

                # If orig is in the GEU, then the dest must also be in it
                if self.materials_by_id[orig] in geu.materials:
                    dest = str(df_geus['Catalogo'][row])

                    geu.graph.add_edge(orig, dest)

            # Set connection strength
            geu.is_strongly_connected = nx.is_strongly_connected(geu.graph)

        # --------------------------------------------------------------------------------------------------------------
        message = "Instantiating GEU objects..."                                                            # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    def create_unigeus(self):
        """
        Create GEU Objects with just one material for those materials with no equivalences
        :param materials_by_id: {id, Material}
        :param geus_by_id: {id, Geu}
        :return: Void -> update GEU dictionary
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        for mat in self.materials_by_id.values():
            if mat.geu is None:
                list_of_materials = list([mat])
                self.geus_by_id[mat.catalog] = GEU(mat.catalog, list_of_materials)
                self.geus_by_id[mat.catalog].is_unigeu = True

                # Make graph
                self.geus_by_id[mat.catalog].graph = nx.DiGraph()
                self.geus_by_id[mat.catalog].graph.add_node(mat.catalog)
                self.geus_by_id[mat.catalog].is_strongly_connected = True

        # --------------------------------------------------------------------------------------------------------------
        message = "Instantiating uniGEUs..."                                                                # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    def complete_geu_information(self, df_lt_spm: pd.DataFrame, df_lt_repair: pd.DataFrame):
        """
        Fill geus information following heuristics described inside the method complete_object_information()
        :param geus_by_id: {id, GEU}
        :param df_lt_spm: pd.DataFrame
        :param df_lt_repair: pd.DataFrame
        :return: Void -> complete GEU Object
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Name override
        query = """SELECT * FROM produccion.geu_name_override;"""
        df_name_override = functions.create_query(query, self.engine)
        df_name_override = df_name_override.astype({'geu': str})
        df_name_override.set_index('geu', inplace=True)

        # Complete GEU information
        [geu.complete_object_information(self.domain_list, df_lt_spm, df_lt_repair,
                                         df_name_override) for geu in self.geus_by_id.values()]

        # Delete df
        del df_name_override

        # Domain override
        query = """SELECT * FROM produccion.validacion_geu_dominio;"""
        df_override = functions.create_query(query, self.engine)
        df_override = df_override.set_index('geu', drop=True)

        for geu in self.geus_by_id.values():
            if geu.catalog in df_override.index:
                geu.domain = df_override.loc[geu.catalog].Validacion
                geu.domain_override = True

            if geu.has_multiple_domains and not geu.domain_override:
                self.geus_with_multiple_domains.append(geu.id)

        if self.is_streamlit:
            if self.geus_with_multiple_domains:
                st.warning(f'Los GEUs {self.geus_with_multiple_domains} contienen múltiples dominios. '
                           f'Se puede cambiar directamente desde la tabla "Validación GEU-Dominio" o '
                           f'"Materiales" o en la sección "Cargar Nuevos Datos"')

        # Complete GEU demands
        for geu in self.geus_by_id.values():
            geu.set_demands()
            geu.update_demand_priorities(self.wh_list, self.start_date)

        # --------------------------------------------------------------------------------------------------------------
        message = "Completing GEUs' info..."                                                                # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    #  ================================================ WH Processing ================================================

    def process_warehouses_leadtimes(self):
        """
        Set leadtime between warehouses
        :return: void -> set static attribute of StockInTransit Class
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Load GREP's province
        if self.grep_type == '7_crams':
            query = """SELECT DISTINCT grep, provincia_grep FROM r00_relation_cram_province;"""
        else:
            query = """SELECT DISTINCT grep, provincia_grep FROM r00_relation_grep_warehouse;"""

        df_grep_province = functions.create_query(query, self.engine)

        dict_grep_province = {grep: province for grep, province in df_grep_province.values}

        # Load leadtimes between provinces
        query = """select * from r00_tiempo_entre_provincias;"""
        df_relation_times = functions.create_query(query, self.engine)

        aux = {(origin, destination): time for origin, destination, time in df_relation_times.values}

        # Generate leadtimes between GREPs
        self.leadtime_between_warehouses = {(wh1, wh2): aux[(dict_grep_province[wh1], dict_grep_province[wh2])]
                                            for wh1 in dict_grep_province.keys()
                                            for wh2 in dict_grep_province.keys()}

        for i in self.wh_list:
            for j in self.wh_list:
                self.df_leadtimes.at[i, j] = self.leadtime_between_warehouses.get((i, j), 4)
        # --------------------------------------------------------------------------------------------------------------
        message = "Set leadtimes between warehouses..."                                                    # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    #  ============================================= Ticket Processing ================================================

    def process_tickets(self, df_tickets):
        """
        Set GEU tickets for simulation tests
        :return: void -> Complete GEU object info
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Add tickets demands for greps without movements demands
        if self.demand_type != 'tickets':
            [self.materials_by_id[str(material)].add_demand(functions.to_datetime(date), grep, int(amount))
             for date, material, amount, grep in df_tickets[["fecha", "material", "cantidad", "grep"]].values
             if str(material) in self.materials_by_id.keys() and
             grep not in set([grep for date, grep, amount in self.materials_by_id[str(material)].demands])]

        # Create Ticket class
        [Ticket(row, self.materials_by_id[str(row["material"])])
         for x, row in df_tickets.iterrows()
         if str(row["material"]) in self.materials_by_id.keys()]

        # --------------------------------------------------------------------------------------------------------------
        message = "Set tickets history..."                                                                  # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    def _set_initial_stocks(self):
        string_date = str(self.start_date.year) + '-' + str(self.start_date.month) + '-' + str(self.start_date.day)

        # Load initial stock
        if self.is_s4:
            query = f"""SELECT material, fecha, ce_alm_s44, stock 
                                                FROM produccion.stock_s4
                                                where fecha = '{string_date}'"""
        else:
            query = f"""SELECT material, fecha, ce_alm_s44, stock 
                                    FROM produccion.stock
                                    where fecha = '{string_date}'"""
        df = functions.create_query(query, self.engine)
        df.columns = ['material', 'fecha', 'grep', 'stock']
        df['fecha'] = pd.to_datetime(df['fecha'])

        # Match with GREP
        greps_banned = [wh for wh in list(df['grep'].unique()) if wh not in self.relation_grep_wh.keys()]
        if self.is_streamlit and greps_banned:
            st.warning(f"Las siguientes relaciones de centro-almacén no se encontraron en la base de datos relacion_grep_almacen: {greps_banned}. "
                       f"No serán tenidos en cuenta. Para que sean incluidos se deben agregar dichas relaciones con su grep correspondiente en relacion_grep_alamcen")
        df = df[~df['grep'].isin(greps_banned)]
        df['grep'] = df['grep'].map(lambda x: self.relation_grep_wh[x])

        # Create material-geu dictionary
        mat_geu_dict = {}
        for mat_id, mat in self.materials_by_id.items():
            mat_geu_dict[mat_id] = mat.geu.id

        # Get material's GEU id
        df['geu'] = df['material'].map(mat_geu_dict)

        # Drop rows without match
        df.dropna(subset=['geu'], inplace=True)
        # Drop material column
        df.drop('material', axis=1, inplace=True)

        """if self.grep_type == 'unicram':
            df['grep'] = 'C00'"""

        # Sum stock by date, grep and geu
        df = df.groupby(['fecha', 'grep', 'geu'])['stock'].sum().reset_index()
        # Turn negative values into 0
        df['stock'] = df['stock'].clip(lower=0)
        #df = df[df['fecha'] == string_date]

        return df

    def set_geu_clusters(self):
        """
        Use sklearn K-Means algorithm in order to reduce complexity of neural domain input
        :param geus_by_id: {id, GEU}
        :return: Void -> complete GEU Object
        """
        tic = functions.elapsed_time_message(True)                                                         # Start Timer
        # --------------------------------------------------------------------------------------------------------------

        # Get variables for clustering
        if self.is_streamlit:
            clustering_date = self.start_date
        else:
            clustering_date = dt.datetime(2020, 1, 2)
        df_for_clustering = data_for_clustering(self.geus_by_id, date_of_clustering=clustering_date)

        # Calculate clusters
        df_clustered_geus, k_clusters = cluster_geus(df_for_clustering)
        # df_clustered_geus.to_csv('df_clustered_geus.csv', index=False)

        # Set clusters
        for index, row in df_clustered_geus.iterrows():
            geu_id = str(row['GEU'])
            self.geus_by_id[geu_id].cluster = int(row['label'])
        self.k_clusters = k_clusters

        # --------------------------------------------------------------------------------------------------------------
        message = "Calculating GEUs' clusters..."                                                           # Stop Timer
        if self.print_on: functions.elapsed_time_message(False, message, tic)
        if self.is_streamlit: functions.elapsed_time_message(False, message, tic,
                                                             is_streamlit=self.is_streamlit)

    def reorder_wh_list(self):
        wh = {wh: 0 for wh in self.wh_list}
        for geu in self.geus_by_id.values():
            for list_of_tickets in geu.tickets.values():
                for ticket in list_of_tickets: wh[ticket.grep] += ticket.amount
        sorted_wh = sorted(wh.items(), key=lambda x: x[1], reverse=True)
        self.wh_list = [wh for (wh, tickets) in sorted_wh]
        if self.grep_type=='greps':
            index_capital = self.wh_list.index('CAPITAL')
            self.wh_list[index_capital] = self.wh_list[0]
            self.wh_list[0] = 'CAPITAL'

        self.wh_dictionary = {self.wh_list.index(wh): wh for wh in self.wh_list}

    #  ================================================ Miscellaneous ================================================

    def return_values(self):
        return self.materials_by_id, self.geus_by_id, self.wh_list, self.engine, self.k_clusters

    def get_info(self):
        if self.print_on:
            print("="*110)
            print('Data Loading Summary\n')
            print(f'Data Base Connection: {self.engine}\n')
            print(f'Number of Materials loaded: {len(self.materials_by_id.keys())}')
            print(f'Number of GEUs generated: {len(self.geus_by_id.keys())}')
            print(f'Number of Clusters: {self.k_clusters}')
            print("="*110)

    def update_progress_bar(self, bar_progress: st, accumulative: int = 0,
                            progress_step: int = 0, is_end: bool = False):
        if bar_progress:
            if is_end:
                bar_progress.progress(100)
            else:
                accumulative = accumulative + progress_step
                bar_progress.progress(accumulative)

                return accumulative

    def get_moving_price(self, weight, travel_time):
        prices = {1: [430, 595],  # [regional, nacional]
                  5: [520, 735],
                  10: [655, 975],
                  15: [795, 1205],
                  20: [955, 1410],
                  25: [1155, 1690]}
        if travel_time <= 3:  # regional
            indice = 0
        else: # nacional
            indice = 1
        last_w = min(prices.keys())
        for w in sorted(prices.keys()):
            if w > weight:
                return prices[last_w][indice]
            last_w = w
        return prices[last_w][indice]

    def save(self, filename: str = 'last_data_model'):
        engine = self.engine
        self.engine = None
        if self.grep_type == '7_crams':
            filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '_cram.pkl')
        elif self.grep_type == 'greps':
            filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '.pkl')
        elif self.grep_type == 'unicram':
            filedir = os.path.join(os.path.dirname(__file__), 'pickles/' + filename + '_unicram.pkl')
        with open(filedir, 'wb') as file:
            pkl.dump(self, file)
        print("Data pickled into file")
        self.engine = engine

    def mat_geu_to_sql(self):
        name = 'produccion.mat_geu'
        self.engine.execute(f"""DELETE FROM {name}""")
        self.insert_new_data_into_sql(df=pd.DataFrame([(mat.catalog, mat.geu.id) for mat in self.materials_by_id.values()],
                                                      columns=['material', 'geu']),
                                      name=name,
                                      columns_types=['String','String'])

    def insert_new_data_into_sql(self, df, name, columns_types):

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

        self.engine.execute(query)
