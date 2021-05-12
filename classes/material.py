from functions import linear_forecast, expo_forecast, to_datetime
import datetime as dt
import re


class Material:
    # Constructors------------------------------------------------------------------------------------------------------

    def __init__(self, catalog, catalog_s4, name, equipment, equipments, domain, factory_code, brand, area, subarea,
                 is_buyable, is_dismountable, is_repairable, is_spm, criticality, price, weight, price_unit,
                        price_rep, rep_unit, tc):

        # Un-inherited Material Data -> directly from input data
        self.domain = domain
        self.brand = brand
        self.name = name
        self.equipment = equipment
        self.equipments = equipments
        self.factory_code = factory_code
        self.area = area
        self.subarea = subarea
        self.catalog = catalog
        self.catalog_s4 = catalog_s4

        # Procurement
        self.is_dismountable = is_dismountable if is_dismountable else False
        self.is_buyable = is_buyable if is_buyable else False
        self.is_spm = is_spm if is_spm else False
        self.is_repairable = is_repairable if is_repairable else False
        self.has_procurement_type = any([self.is_dismountable, self.is_buyable, self.is_spm, self.is_repairable])

        # Price and price flags
        self.tc = tc
        self.price = None
        self.price_type = None
        self.rep_price = None
        self.price_unit = price_unit
        self.rep_unit = rep_unit
        self.rep_price_type = None
        if price is not None:
            self.set_price(price, price_type='3-Known')
        if price_rep is not None:
            self.set_rep_price(price_rep, price_type='3-Known')

        # Criticality
        self.criticality = None
        self.has_criticality_data = False
        if criticality is not None:
            self.set_criticality(criticality)

        # Weight
        try:
            self.weight = float(weight)
        except:
            self.weight = None

        # GEU
        self.is_unigeu = False
        self.geu = None
        self.equivalent_materials = set()

        # Lead Time
        self.leadtime = None
        self.leadtime_sd = None

        self.demands = []
        self.stock = {}
        self.movements = {}
        self.warehouses = set()

        self.starting_stock = {}
        self.has_starting_stock = False

        self.tickets = {}

    # Constructor from Data frame columns
    @staticmethod
    def create_material_from_df(material_df_row):
        # Remove equipment unnecessary spaces
        equipment = material_df_row["Equipo / Modelo"]
        remove_ws = re.compile(r'\s+')
        equipment = remove_ws.sub(' ', equipment).strip(' \t\n\r')

        equipments = []
        # Different equipments (if applies)
        if equipment.find('/') != -1:
            limit = 9
            for i in range(0, limit):
                # 0000/0000/... pattern
                decoder = re.compile(r'(.*)(\d{4})' + (limit - i) * '/(\d{4})' + '(.*)')
                matches = decoder.findall(equipment)

                # If we have a perfect match, take equipment modified names and leave for loop
                if matches:
                    for j in range(2, limit + 3 - i):
                        equipments.append(decoder.sub(r'\1' + f'\\{j}' + f'\\{limit + 3 - i}', equipment))
                    break

                # 0000/00/00/... pattern
                decoder = re.compile(r'(.*)(\d{4})' + (limit - i) * '/(\d{2})' + '(.*)')
                matches = decoder.findall(equipment)

                # If we have a perfect match, take equipment modified names and leave for loop
                if matches:
                    # Append first equipment
                    match_prefix = matches[0][1][0:2]
                    equipments.append(decoder.sub(r'\1\2' + f'\\{limit + 3 - i}', equipment))
                    # Append rest
                    for j in range(3, limit + 3 - i):
                        equipments.append(decoder.sub(r'\1', equipment) + match_prefix +
                                          decoder.sub(r'' + f'\\{j}' + f'\\{limit + 3 - i}', equipment))
                    break
        # If there was no pattern, or if the specific patterns just weren't there, append the only equipment
        if not equipments:
            # Just append the equipment
            equipments.append(equipment)

        return Material(catalog=str(material_df_row["Catalogo"]),
                        catalog_s4=str(material_df_row["Catalogo_S4"]),
                        name=material_df_row["Descripcion SAP"],
                        equipment=equipment,
                        equipments=equipments,
                        domain=material_df_row["Dominio"],
                        factory_code=material_df_row["CodFabr"],
                        brand=material_df_row["Marca"],
                        area=material_df_row["area"],
                        subarea=material_df_row["SubAREA"],
                        is_buyable=material_df_row["Comprable"],
                        is_dismountable=material_df_row["Desmontable"],
                        is_repairable=material_df_row["Reparable"],
                        is_spm=material_df_row["spm"],
                        criticality=material_df_row['Criticidad'],
                        price=material_df_row["Precio"],
                        weight=material_df_row["Peso Kg"],
                        price_unit=material_df_row["Unidad_Precio"],
                        price_rep=material_df_row["Costo_Reparaciones"],
                        rep_unit=material_df_row["Unidad_Reparacion"],
                        tc=material_df_row["TC"])

    # ==================================================================================================================

    def __repr__(self):
        return "<Material> id:{} marca:{}".format(self.catalog, self.brand)

    # Setters-----------------------------------------------------------------------------------------------------------

    def set_criticality(self, criticality):
        criticality = criticality.lower()

        if criticality in ['critico', 'critica', 'alto', 'alta']:
            self.criticality = 'critico'
            self.has_criticality_data = True
        elif criticality in ['mayor', 'medio', 'media']:
            self.criticality = 'mayor'
            self.has_criticality_data = True
        elif criticality in ['bajo', 'baja']:
            self.criticality = 'bajo'
            self.has_criticality_data = True
        else:
            print(f"Error - couldn't set criticality for material {self}")
            print("Criticality must take the following values:")
            print("[['critico', 'alta', 'alto'], ['mayor', 'media', 'medio'],", end=' ')
            print(f"['bajo', 'baja']] and {criticality} was given")
            print(f'Se le asigna criticidad Baja')
            self.criticality = 'bajo'

    def set_geu(self, geu, list_of_materials):
        self.geu = geu
        for material in list_of_materials:
            self.equivalent_materials.add(material)

    def set_leadtimes(self, mean_leadtime, standard_deviation_leadtime):
        self.leadtime = mean_leadtime
        self.leadtime_sd = standard_deviation_leadtime

    def set_price(self, price, price_type: str):
        if price > 0:
            if self.price_unit in ['usd', 'USD']:
                self.price = price
            else:
                self.price = int(price / self.tc)
        self.price_type = price_type
        self.price_unit = "USD"

    def set_rep_price(self, price_rep, price_type: str):
        if price_rep > 0:
            if self.rep_unit in ['usd', 'USD']:
                self.rep_price = price_rep*self.tc
            else:
                self.rep_price = price_rep
        self.rep_price_type = price_type
        self.rep_unit = "ARS"

    def set_stock(self, date, warehouse, amount):
        self.stock[(date, warehouse)] = amount
        self.warehouses.add(warehouse)

    def set_movement(self, date, warehouse, amount):
        self.warehouses.add(warehouse)
        self.movements[(to_datetime(date), warehouse)] = amount

    def set_starting_stock(self, warehouse, amount):
        self.cstarting_stk[warehouse] = amount
        self.has_starting_stock = True

    def set_ticket(self, ticket):

        if ticket.ticket_date not in self.tickets.keys():
            self.tickets[ticket.ticket_date] = []
            self.tickets[ticket.ticket_date].append(ticket)
            self.geu.set_ticket(ticket)
        else:
            self.tickets[ticket.ticket_date].append(ticket)
            self.geu.set_ticket(ticket)

    def set_movements_for_teco_simulation(self, date, cram, amount):
        self.geu.set_movements_for_teco_simulation(date, cram, amount)

    # ==================================================================================================================

    # Getters-----------------------------------------------------------------------------------------------------------

    def get_demand(self, period_object, warehouse='all'):
        if warehouse == 'all':
            list_of_amounts = [amount for date, ware_house, amount in self.demands if period_object.contains(date)]
        else:
            list_of_amounts = [amount for date, ware_house, amount in self.demands if period_object.contains(date) and
                               ware_house == warehouse]

        return sum(list_of_amounts)

    def add_demand(self, date, warehouse, demand):
        self.demands.append([date, warehouse, demand])
        self.warehouses.add(warehouse)

    def get_forecast(self, period_object, warehouse, time_delta=dt.timedelta(days=365 * 3), soft=False, type='linear'):

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

    def get_first_date_demand(self, warehouse):
        if self.demands:
            if warehouse == 'all':
                return min([dates for dates, warehouse, demand in self.demands])
            else:
                list_of_demands = [dates for dates, wh, demand in self.demands if wh == warehouse]
                return min(list_of_demands) if len(list_of_demands) != 0 else None

    def get_stock(self, date, warehouse):
        return self.stock[(date, warehouse)]
    # ==================================================================================================================
