import numpy as np
import pandas as pd
import itertools
from math import pi
from datetime import datetime, timedelta
from typing import Union
from tqdm import tqdm
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler
from pylab import rcParams
import seaborn as sns

from classes.material import Material
from classes.geu import GEU
from classes.period import Period
from functions import create_query, upload_from_dataframe

# Color palette
acn_colors = ['#a100ff', '#7600c0', '#460073',
              '#00baff', '#008eff', '#004dff',
              '#969696', '#5a5a5a', '#000000']
rcParams['axes.prop_cycle'] = cycler(color=acn_colors)


def validation_data(group: Union[GEU, Material]):
    both_vars = ['Catalog', 'Price', 'Price Type', 'Buyable',
                 'Criticality', 'Has Criticality Data', 'Domain', 'Weight']
    geu_vars = ['Cluster', 'Weighted Demand v1 (TKT)', 'Leadtime', 'Procurement Type']
    mat_vars = ['SPM', 'Repairable', 'Dismountable', 'Has Procurement Type']

    crit_dict = {'bajo': 0, 'mayor': 1, 'critico': 2}

    # Take class
    for item in group.values():
        item_class = type(item)
        break

    # Add appropiate variables
    var_cols = both_vars
    if item_class == GEU:
        var_cols += geu_vars
    elif item_class == Material:
        var_cols += mat_vars
    else:
        raise Exception('Class not allowed for validation')

    # Create variables list of lists
    variables = []
    for i in range(len(var_cols)):
        variables.append([])

    # Append values and increase index
    def append_new(index, variable_lol, variable):
        variable_lol[index].append(variable)
        index += 1
        return index

    # Fill lists
    for item in group.values():
        i = 0
        i = append_new(i, variables, item.catalog)
        i = append_new(i, variables, item.price)
        i = append_new(i, variables, item.price_type)
        i = append_new(i, variables, item.is_buyable)
        i = append_new(i, variables, crit_dict[item.criticality] if item.criticality is not None else None)
        i = append_new(i, variables, item.has_criticality_data)
        i = append_new(i, variables, item.domain)
        i = append_new(i, variables, item.weight)

        if item_class == GEU:
            i = append_new(i, variables, item.cluster)
            i = append_new(i, variables, item.weighted_demand_v1_tkt)
            i = append_new(i, variables, item.leadtime)
            i = append_new(i, variables, item.procurement_type)

        elif item_class == Material:
            i = append_new(i, variables, item.is_spm)
            i = append_new(i, variables, item.is_repairable)
            i = append_new(i, variables, item.is_dismountable)
            i = append_new(i, variables, item.has_procurement_type)

    # Make df
    df = pd.DataFrame(variables)
    df = df.transpose()
    df.columns = var_cols

    # Other variables

    # Information
    if item_class == Material:
        df.loc[df['Buyable'], "Procurement Type 1"] = 1
        df.loc[df['SPM'], "Procurement Type 1"] = 2
        df.loc[df['Dismountable'], "Procurement Type 1"] = 3
        df.loc[~(df['Buyable'] | df['SPM'] | df['Dismountable']), "Procurement Type 1"] = 0
        df['Procurement Type 2'] = df['Repairable'].apply(lambda x: 1 if x else 0)

    # Criticality Data
    def raw_crit(row):
        crit_dict_inv = {v: k for k, v in crit_dict.items()}
        if row['Has Criticality Data']:
            val = crit_dict_inv[row['Criticality']]
        else:
            val = 'Sin criticidad'
        return val
    df['Raw Criticality'] = df.apply(raw_crit, axis=1)

    return df


def locate_figure(win_w, win_h, where: str = 'left'):
    # Put figure in the corresponding corner
    mngr = plt.get_current_fig_manager()
    if where == 'left':
        mngr.window.geometry(f'{win_w // 2}x{win_h}+0+0')
    elif where == 'right':
        mngr.window.geometry(f'{win_w // 2}x{win_h}+{win_w // 2}+0')
    elif where == 'full':
        mngr.window.geometry(f'{win_w}x{win_h}+0+0')


def validation(df, clusters: int = 0, win_w: int = 1280, win_h: int = 1080, locate_figures: bool = False):
    # Close related plots
    for i in plt.get_fignums():
        if plt.figure(i).get_label()[0:5] == "(VAL)":
            plt.close(plt.figure(i).get_label())

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} ({p:.0f}%)'.format(p=pct, v=val)

        return my_autopct

    def draw_pie(figure: int, series: pd.Series, variable: str, sorted_colors: bool = False,
                 where: str = 'left', cdict: dict = None, text_color: str = 'white'):
        # Figure
        plt.figure(f'(VAL) {figure} Pie {variable}')
        plt.title(f'Distribución de {variable}', fontsize=18)

        # Put figure in the corresponding corner
        if locate_figures:
            locate_figure(win_w, win_h, where=where)

        # --------------

        # Sorted ids
        labels, counts = np.unique(series, return_counts=True)
        sorted_ids = np.argsort(-counts)

        # Apply: cdict, sorted_colors, or neither
        if cdict:
            replaced_labels = labels[sorted_ids]
            for i in range(len(replaced_labels)):
                if replaced_labels[i] not in cdict.keys():
                    replaced_labels[i] = 'Other'

            _, _, autotexts = plt.pie(counts[sorted_ids],
                                      labels=labels[sorted_ids], autopct=make_autopct(counts[sorted_ids]),
                                      colors=[cdict[domain] for domain in replaced_labels],
                                      startangle=0, wedgeprops={'edgecolor': 'black'})
        elif sorted_colors:
            _, _, autotexts = plt.pie(counts[sorted_ids],
                                      labels=labels[sorted_ids], autopct=make_autopct(counts[sorted_ids]),
                                      colors=[acn_colors[key] for key in sorted_ids],
                                      startangle=0, wedgeprops={'edgecolor': 'black'})
        else:
            _, _, autotexts = plt.pie(counts[sorted_ids],
                                      labels=labels[sorted_ids], autopct=make_autopct(counts[sorted_ids]),
                                      startangle=0, wedgeprops={'edgecolor': 'black'})

        # Set text color
        for autotext in autotexts:
            autotext.set_color(text_color)

    def draw_histogram(figure: int, series: pd.Series, variable: str, where: str = 'left',
                       max_val: int = 3000, money: bool = True):
        # Figure
        plt.figure(f'(VAL) {figure} Hist. {variable}')
        if locate_figures:
            locate_figure(win_w, win_h, where=where)

        # Subplot 1: Histogram as-is
        plt.subplot(211)
        title = f'Distribución de {variable}'
        if money:
            title += ' en USD'
        plt.title(title, fontsize=18)

        # Subplot 1: Histogram as-is
        plt.hist(series, edgecolor='black', linewidth=0.8)

        # Axis
        plt.gca().set(ylabel='Frecuencia')
        if money:
            fmt = '${x:,.0f}'
            tick = mtick.StrMethodFormatter(fmt)
            plt.gca().xaxis.set_major_formatter(tick)

        # Subplot 2: Zoomed for range 0-max_val
        plt.subplot(212)
        plt.hist(series, edgecolor='black', linewidth=0.8, range=[0, max_val])

        # Axis
        xlabel = str(variable)
        if money:
            xlabel += ' [US$]'
        plt.gca().set(xlabel=xlabel,
                      ylabel='Frecuencia')
        if money:
            fmt = '${x:,.0f}'
            tick = mtick.StrMethodFormatter(fmt)
            plt.gca().xaxis.set_major_formatter(tick)
        plt.tight_layout()

    def draw_boxplot(cluster: int, data, edge_color, fill_color):
        # Make boxplot
        bp = ax.boxplot(data, showfliers=False, patch_artist=True, positions=[cluster])

        # All the rest
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        # Box color
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

    def draw_radarplot(figure: int, df: pd.DataFrame, feats: dict, group_type: str,
                       clusters: int = 6, where: str = 'left'):
        # PREPROCESSING

        # Group variable
        group = df.columns[0]

        # Features
        categories = list(df)[1:]
        categories_spa = []
        for cat in categories:
            try:
                categories_spa.append(feats[cat][0])
            except:
                categories_spa.append('Cercanía Último Ticket')
        N = len(categories)

        # ------- PART 1: Create background

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialize the spider plot
        plt.figure('(VAL) ' + str(figure) + ' Radar clusters')
        ax = plt.subplot(111, polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories_spa)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(list(np.arange(0, 1.2, 0.2)), color="grey", size=7)
        plt.ylim(0, 1)

        # ------- PART 2: Add plots

        # Ind1
        for i in range(clusters):
            values = df.loc[i].drop(group).values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'{group_type} {i}')
            ax.fill(angles, values, alpha=0.1)

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.2))

        # Put figure in the corresponding corner
        if locate_figures:
            locate_figure(win_w, win_h, where=where)

    def draw_bubbleplot(figure: int, df: pd.DataFrame, raw: bool = False, where: str = 'left'):
        # Create a figure instance
        if raw:
            fig = plt.figure(f'(VAL) {figure} Tipo Abastecimiento (Crudo)')
            plt.title('Tipo de Abastecimiento (Raw)', fontsize=18)
        else:
            fig = plt.figure(f'(VAL) {figure} Tipo Abastecimiento')
            plt.title('Tipo de Abastecimiento', fontsize=18)

        # Put figure in the corresponding corner
        if locate_figures:
            locate_figure(win_w, win_h, where=where)

        # Create an axes instance
        ax = fig.add_subplot(111)

        # Axis
        plt.gca().set(xlabel='Tipo de abastecimiento',
                      ylabel='Reparable')
        ax.set_xlim([-0.5, 3.5])
        ax.set_ylim([-0.5, 1.5])

        # Make aggregated df
        df_procurement = df[['Procurement Type 1', 'Procurement Type 2', 'Catalog', 'Has Procurement Type']]
        if raw:
            df_procurement = df_procurement[df_procurement['Has Procurement Type']]
        df_procurement = df_procurement.groupby(['Procurement Type 1', 'Procurement Type 2']).count()
        df_procurement.rename({'Catalog': 'Catalogs'}, axis=1, inplace=True)
        df_procurement = df_procurement.astype({'Catalogs': int})
        df_procurement.reset_index(inplace=True)

        # Scatter plot
        plt.scatter(df_procurement['Procurement Type 1'], df_procurement['Procurement Type 2'],
                    s=df_procurement['Catalogs'], linewidths=0.5, edgecolors='black')

        # Set tick frequency
        loc = mtick.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        loc = mtick.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
        ax.yaxis.set_major_locator(loc)

        # Set text axis
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[1] = 'Ninguno'
        labels[2] = 'Comprable'
        labels[3] = 'SPM'
        labels[4] = 'Desmontable'
        ax.set_xticklabels(labels)

        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[1] = 'No'
        labels[2] = 'Sí'
        ax.set_yticklabels(labels)

        # Plot labels
        for index, row in df_procurement.iterrows():
            ax.annotate(int(row['Catalogs']), (row['Procurement Type 1'], row['Procurement Type 2']),
                        ha='center', va='center')

        # If raw, plot errors in red
        if raw:
            df_procurement = df['Has Procurement Type']
            no_info_count = df_procurement.shape[0] - df_procurement.sum()
            plt.scatter(0, 0, s=no_info_count,
                        linewidths=0.5, edgecolors='black', c='red')
            ax.annotate(no_info_count, (0, 0), ha='center', va='center')

    def make_violin(figure: int, x: pd.Series, y: pd.Series, where: str = 'left'):
        # Create a figure instance
        fig = plt.figure(f'(VAL) {figure} Leadtime Tipo Abastecimiento')
        plt.title('Leadtime por Tipo de Abastecimiento', fontsize=18)

        # Put figure in the corresponding corner
        if locate_figures:
            locate_figure(win_w, win_h, where=where)

        # Make violinplot
        sns.violinplot(x=x, y=y)

    # Initialize fig_num
    fig_num = 1

    # ----------------------------------------------------

    # High level information

    if 'Domain' in df.columns:
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

        draw_pie(figure=fig_num, series=df['Domain'], variable='Dominio', where='left',
                 cdict=colors)
        fig_num += 1
    if 'Raw Criticality' in df.columns:
        draw_pie(figure=fig_num, series=df['Raw Criticality'],
                 variable='Información de Criticidad', where='right',
                 cdict={'bajo': acn_colors[0], 'mayor': acn_colors[1], 'critico': acn_colors[2], 'Other': acn_colors[3]})
        fig_num += 1

    # -------------

    # Procurement and Price

    # Procurement
    if 'Has Procurement Type' in df.columns:
        draw_bubbleplot(fig_num, df, raw=True, where='left')
        fig_num += 1

    # Price
    if 'Price' in df.columns:
        draw_histogram(figure=fig_num, series=df[df['Price'].notnull()]['Price'], variable='Precio', where='left')
        fig_num += 1

        # Only prices for buyable materials
        draw_histogram(figure=fig_num, series=df[df['Buyable']]['Price'],
                       variable='Precio de compra', where='right')
        fig_num += 1

    # Weight
    draw_histogram(figure=fig_num, series=df[df['Weight'].notnull()]['Weight'],
                   variable='Peso', where='left', max_val=10, money=False)
    fig_num += 1

    # Price Type
    colors = {'0-Mean': acn_colors[0],
              '1-Domain': acn_colors[1],
              '2-Model': acn_colors[2],
              '3-Known': acn_colors[3],
              'Other': 'lime'}
    if 'Price Type' in df.columns:
        draw_pie(figure=fig_num, series=df['Price Type'], variable='Tipo de Precio',
                 sorted_colors=True, where='left', cdict=colors)
        fig_num += 1

        if 'Buyable' in df.columns:
            # Buyable Price Type
            draw_pie(figure=fig_num, series=df[df['Buyable']]['Price Type'], variable='Tipo de Precio de Compra',
                     sorted_colors=True, where='right', cdict=colors)
            fig_num += 1

    # Leadtimes
    if 'Leadtime' in df.columns:
        make_violin(figure=fig_num, x=df['Procurement Type'], y=df['Leadtime'].astype(int))
        fig_num += 1

    # -------------

    # Clustering

    # Pie chart
    if 'Cluster' in df.columns:
        draw_pie(figure=fig_num, series=df['Cluster'], variable='Cluster', sorted_colors=True, where='left')
        fig_num += 1


def stock_metrics(df: pd.DataFrame, geus: dict, ids: Union[str, list],
                  plot: bool = False, calculation_mode: bool = False, locate_figures: bool = False,
                  last_year: int = 2019):
    print('Calculating stock metrics...')

    # Close related plots
    if plot:
        for i in plt.get_fignums():
            if plt.figure(i).get_label()[1:5] == "STK2":
                plt.close(plt.figure(i).get_label())

        # Window properties
        win_w = 1280
        win_h = 1080

    min_date = df['fecha'].min()
    max_date = df['fecha'].max()

    # Calculate stock means
    df_mean = df.groupby(['anio', 'geu']).mean()

    # Get periods
    periods = []
    go_on = True
    set_period = Period(datetime(min_date.year, 1, 1), 'Y', 1)
    while go_on:
        if (set_period.last_date < max_date or set_period.contains(max_date))\
                and set_period.last_date.year <= last_year:
            periods.append(set_period)
            set_period = set_period.next()
        else:
            go_on = False

    # Preprocess ids
    if type(ids) == str:
        if ids == 'all':
            ids = list(geus.keys())
        else:
            ids = [ids]

    # Initialize indicators
    if calculation_mode:
        geus_indicators = {'geu': None, '2018': None, '2019': None, 'Demand Types Ratio': None}
        df_stock_info = pd.DataFrame(columns=list(geus_indicators.keys()))

    def indicators(id, geu_index: int = None):
        # Print info only if needed
        if calculation_mode:
            geus_indicators['index'] = geu_index
            geus_indicators['geu'] = id
        else:
            geus[id].get_info()
            print()

        # Calculate stock & demand indicators for each period
        for period in periods:
            # Demand
            year = period.first_date.year
            demand = geus[id].get_demand(period, 'all')
            if period.contains(min_date):
                days = (period.last_date - min_date).days + 1
            elif period.contains(max_date):
                days = (max_date - period.first_date).days + 1
            else:
                days = period.days_elapsed()
            annual_demand = demand * 365.25 / days

            if calculation_mode:
                period_values = []
                period_values += [year, annual_demand]

            # Average stock
            no_stock = True
            if not df_mean.empty:
                try:
                    avg_stock = df_mean.loc[year, id][0]
                    no_stock = False
                    if calculation_mode:
                        period_values.append(avg_stock)
                except:
                    if calculation_mode:
                        period_values.append('')

            # Print demand & stock indicators (if applies)
            if calculation_mode:
                geus_indicators[str(period.first_date.year)] = period_values
            else:
                print(year)
                print(f"{demand} units in {days} days")
                print('Annualized demand:', round(annual_demand, 2))
                if no_stock:
                    print('No stock was found in the database')
                else:
                    print('Average stock:', round(avg_stock, 2))
                    print('Avg. stock in years of demand:', round(avg_stock / annual_demand, 2))
                print()

        # -------------

        # Tickets
        ticket_demands = []
        for key, tickets in geus[id].tickets.items():
            ticket_demands.append([key, sum([ticket.amount for ticket in tickets])])
        df_tickets = pd.DataFrame(ticket_demands, columns=['date', 'ticketed'])
        df_tickets['date'] = pd.to_datetime(df_tickets['date']).dt.date
        df_tickets = df_tickets[df_tickets['date'].apply(lambda x: x.year) <= last_year]

        # Demands
        demands = {}
        for row in geus[id].demands:
            if row[0] not in demands:
                demands[row[0]] = 0
            demands[row[0]] += row[2]
        df_demands = pd.DataFrame.from_dict(demands, orient='index', columns=['demanda'])
        df_demands.reset_index(inplace=True)
        df_demands.rename({'index': 'date'}, axis=1, inplace=True)
        df_demands['date'] = pd.to_datetime(df_demands['date']).dt.date
        df_demands = df_demands[df_demands['date'].apply(lambda x: x.year) <= last_year]

        # Plot ticket demand vs geu demand
        if plot:
            # Same axis
            plt.figure(f"(STK2) Tickets y Demandas - GEU {id}")
            plt.title(f"Tickets vs Demanda - GEU {id}", fontsize=18)
            if ticket_demands:
                plt.bar(df_tickets['date'], df_tickets['ticketed'], label='Tickets',
                        width=timedelta(days=1), color='blue')
            if demands:
                plt.bar(df_demands['date'], df_demands['demanda'], label='Demandas',
                        width=timedelta(days=1), color='brown')
            plt.legend()
            if locate_figures:
                locate_figure(win_w, win_h, where='full')

            # Two axis
            if ticket_demands and demands:
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                               num=f"(STK2) Tickets vs Demandas - GEU {id}")
                ax1.set_title(f"Tickets vs Demanda - GEU {id}", fontsize=18)
                ax1.bar(df_tickets['date'], df_tickets['ticketed'], label='Tickets',
                        width=timedelta(days=1), color='blue')
                ax1.legend()
                ax2.bar(df_demands['date'], df_demands['demanda'], label='Demandas',
                        width=timedelta(days=1), color='brown')
                ax2.legend()
                if locate_figures:
                    locate_figure(win_w, win_h, where='full')

        # -------------

        # Ticket validation
        df_merge = df_demands.merge(df_tickets, how='outer').sort_values('date')
        if demands:
            df_merge['demanda_cumnull'] = df_merge['demanda'].isnull().cumsum()
        if ticket_demands:
            df_merge['ticketed_cumnull'] = df_merge['ticketed'].isnull().cumsum()
        if demands and ticket_demands:
            df_merge = df_merge[(df_merge['demanda_cumnull'] > 0) & (df_merge['ticketed_cumnull'] > 0)]
        df_merge.fillna(0, inplace=True)
        if demands:
            df_merge.drop('demanda_cumnull', axis=1, inplace=True)
            df_merge['demanda_cum'] = df_merge['demanda'].cumsum()
        if ticket_demands:
            df_merge.drop('ticketed_cumnull', axis=1, inplace=True)
            df_merge['ticketed_cum'] = df_merge['ticketed'].cumsum()

        # Plot cummulative ticket demand & geu demand
        if plot:
            # Figure
            plt.figure(f'(STK2) Demanda Acumulada - GEU {id}')
            plt.title(f"Tickets vs Demanda - Acumulado - GEU {id}", fontsize=18)

            # Time series
            if ticket_demands:
                plt.plot_date(df_merge['date'], df_merge['ticketed_cum'], label='Tickets',
                              linestyle='solid', marker='', color='blue')
            if demands:
                plt.plot_date(df_merge['date'], df_merge['demanda_cum'], label='Demandas',
                              linestyle='solid', marker='', color='brown')
            plt.legend()
            if locate_figures:
                locate_figure(win_w, win_h, where='full')

        # More general info
        try:
            if ticket_demands and demands and df_merge.iloc[-1]['ticketed_cum'] > 0:
                demand_types_ratio = df_merge.iloc[-1]['demanda_cum'] / df_merge.iloc[-1]['ticketed_cum']
                if calculation_mode:
                    geus_indicators['Demand Types Ratio'] = demand_types_ratio
                else:
                    print(f"Ratio de Demanda (Mov 351 / Tickets): {round(demand_types_ratio, 2)}")
            else:
                raise Exception()
        except:
            if not calculation_mode:
                print("No es posible calcular el ratio de demanda (Mov 351 / Tickets)")

    total_geus = len(ids)
    geu_index = 0

    # Use indicators method and use progress bar (if applies)
    if calculation_mode:
        with tqdm(total=total_geus, file=sys.stdout) as pbar:
            for id in ids:
                # Run indicators method
                indicators(id, geu_index)
                df_stock_info = df_stock_info.append(geus_indicators, ignore_index=True)
                geus_indicators = {'geu': None, '2018': None, '2019': None, 'Demand Types Ratio': None}
                geu_index += 1

                # Manually update progress bar
                pbar.update(1)

        df_stock_info.reset_index(drop=True, inplace=True)

        return df_stock_info
    else:
        for id in ids:
            indicators(id)


def load_geu_stock(ids: Union[str, list], conn, plot: bool = False, locate_figures: bool = False):
    print('Fetching information from database...')

    # Close related plots
    if plot:
        for i in plt.get_fignums():
            if plt.figure(i).get_label()[1:5] == "STK1":
                plt.close(plt.figure(i).get_label())

        # Window properties
        win_w = 1280
        win_h = 1080

    # Set query
    if type(ids) == str and ids != 'all':
        ids = [ids]
    if ids == 'all':
        query = f"""SELECT geu, fecha, stock, anio FROM p04_geu_stock;"""
    else:
        str_ids = str(ids)[1:-1]
        query = f"""SELECT geu, fecha, stock, anio FROM p04_geu_stock WHERE geu in ({str_ids});"""

    # Get data
    df_stock = create_query(query, conn)
    if df_stock.empty:
        print(f'No stock found for GEUs {ids}')
        return None
    df_stock.reset_index(drop=True, inplace=True)
    df_stock["fecha"] = pd.to_datetime(df_stock["fecha"])
    df_stock.sort_values('fecha', inplace=True)

    # Plot (if applies)
    if plot:
        # Check if plotting many GEUs
        if len(ids) == 1:
            plt.figure(f"(STK1) Stock - GEU {ids[0]}")
            plt.title(f"Stock GEU {ids[0]}", fontsize=18)
        else:
            plt.figure("(STK1) Stock GEUs")
            plt.title("Stock GEUs", fontsize=18)
        if locate_figures:
            locate_figure(win_w, win_h, where='full')

        # Plot time series
        for id in ids:
            df_cut = df_stock[df_stock['geu'] == id]
            plt.plot_date(df_cut['fecha'], df_cut['stock'], linestyle='solid', marker='', label=id)
        plt.legend()

    return df_stock


def geu_report(geus: dict, ids: Union[str, list], conn, last_year: int = 2019,
               plot: bool = False, locate_figures: bool = False, calculation_mode: bool = False):
    # Override plot boolean if using 'all'
    if plot and ids == 'all':
        print("Can't plot all GEUs! Setting plot=False...")
        plot = False

    # Load data
    df_stock = load_geu_stock(ids=ids, conn=conn, plot=plot, locate_figures=locate_figures)
    # Metrics
    df_stock_info = stock_metrics(df=df_stock, geus=geus, ids=ids, last_year=last_year,
                                  plot=plot, locate_figures=locate_figures, calculation_mode=calculation_mode)

    return df_stock_info


def geu_procurements(geus: dict, print_on: bool = True):
    """
    Calculate procurement-related useful DataFrames
    :param geus: dict
    :param print_on: bool
    :return: pd.DataFrame, pd.DataFrame
    """

    if print_on: print("Fetching procurement info...")
    keys_list = ['GEU', 'Cluster', 'Descripción', 'Abastecimiento Final',
                 'SPM/RMA', 'Desmontable', 'Reparable', 'Comprable', 'Catálogos']
    df_procurement = pd.DataFrame(columns=keys_list)

    def add_geu(df, geu):
        # Fetch GEUs catalogs
        materials = []
        for mat in geu.materials:
            materials.append(mat.catalog)

        # Fetch info
        values_list = [geu.catalog, geu.cluster, geu.name, geu.procurement_type,
                       geu.is_spm, geu.is_dismountable, geu.is_repairable, geu.is_buyable, materials]

        # Make df with procurement info
        df = df.append(dict(zip(keys_list, values_list)), ignore_index=True)
        return df

    # Use tqdm progress bar (if applies)
    if print_on:
        with tqdm(total=len(geus), file=sys.stdout) as pbar:
            for geu in geus.values():
                df_procurement = add_geu(df_procurement, geu)

                # Update GEUs progress bar
                pbar.update(1)
    else:
        for geu in geus.values():
            df_procurement = add_geu(df_procurement, geu)

    if print_on: print("Counting unique combinations by cluster...")
    # Get one-hot encoding of cluster column
    df_clusters = df_procurement.drop(['GEU', 'Descripción', 'Catálogos'], axis=1)
    one_hot = pd.get_dummies(df_clusters['Cluster'], prefix='Cluster')
    df_clusters = df_clusters.join(one_hot)
    df_clusters.drop('Cluster', axis=1, inplace=True)

    # Group by procurement types
    df_clusters = df_clusters.groupby(
        ['Abastecimiento Final', 'SPM/RMA', 'Desmontable', 'Reparable', 'Comprable']).sum()

    # Order df by cluster 1 descending
    if print_on: print("Sorting by Cluster 1 subtotals...")
    df_clusters.sort_values('Cluster_1', ascending=False, inplace=True)

    # Print df
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(df_clusters)
    # pd.reset_option('max_columns')
    # pd.reset_option('max_rows')

    return df_procurement, df_clusters


def cluster_and_domain(geus: dict):
    keys_list = ['cluster', 'domain', 'demand']
    df_geu_agg = pd.DataFrame(columns=keys_list)
    for geu in geus.values():
        values_list = [geu.cluster, geu.domain, geu.weighted_demand_v1_tkt]
        df_geu_agg = df_geu_agg.append(dict(zip(keys_list, values_list)), ignore_index=True)

    df_geu_agg = df_geu_agg.astype({'demand': int})
    df_geu_agg = df_geu_agg.groupby(['cluster', 'domain']).agg({'demand': ['count', 'mean', 'std']})
    df_geu_agg.to_csv('df_geu_agg.csv')


def make_relevant_dicts(geus: dict, mats: dict, print_on: bool = True):
    # Relative path
    relative_path = f'exportables/'

    # ---------------------

    if print_on: print('Fetching materials info...')

    # MATERIALS
    keys_list = ['material', 'geu']
    df = pd.DataFrame(columns=keys_list)

    # Iterate through all materials
    if print_on:
        with tqdm(total=len(mats), file=sys.stdout) as pbar:
            for mat in mats.values():
                # Fetch values
                values_list = [mat.catalog, mat.geu.catalog]

                # Make df with material info
                df = df.append(dict(zip(keys_list, values_list)), ignore_index=True)

                # Update bar
                pbar.update(1)
    else:
        for mat in mats.values():
            # Fetch values
            values_list = [mat.catalog, mat.geu.catalog]

            # Make df with material info
            df = df.append(dict(zip(keys_list, values_list)), ignore_index=True)

    # Export df
    df.to_csv(os.path.join(os.path.dirname(__file__), relative_path + 'p01_mat_geu.csv'), index=False)

    if print_on: print('Material info exported.')

    # ---------------------

    if print_on: print('Fetching GEU info...')

    # GEUS
    keys_list = ['geu', 'cluster']
    df = pd.DataFrame(columns=keys_list)
    keys_list_dict = ['geu', 'cluster', 'descripcion', 'dominio', 'subarea', 'marca', 'equipo',
                      'completo', 'precio', 'tipo de precio', 'leadtime',
                      'criticidad', 'tipo de abastecimiento']
    df_dict = pd.DataFrame(columns=keys_list_dict)

    # Iterate through all GEUs
    if print_on:
        with tqdm(total=len(geus), file=sys.stdout) as pbar:
            for geu in geus.values():
                # Fetch values
                values_list = [geu.id, geu.cluster]
                # Make df with GEU info
                df = df.append(dict(zip(keys_list, values_list)), ignore_index=True)

                # Fetch values
                values_list_dict = [geu.id, geu.cluster, geu.name, geu.domain, geu.subarea, geu.brand, geu.equipment,
                                    geu.is_strongly_connected, geu.price, geu.price_type, geu.leadtime,
                                    geu.criticality, geu.procurement_type]
                # Make df with GEU info
                df_dict = df_dict.append(dict(zip(keys_list_dict, values_list_dict)), ignore_index=True)

                # Update bar
                pbar.update(1)
    else:
        for geu in geus.values():
            # Fetch values
            values_list = [geu.id, geu.cluster]
            # Make df with GEU info
            df = df.append(dict(zip(keys_list, values_list)), ignore_index=True)

            # Fetch values
            values_list_dict = [geu.id, geu.cluster, geu.name, geu.domain, geu.subarea, geu.brand, geu.equipment,
                                geu.price, geu.criticality, geu.procurement_type]
            # Make df with GEU info
            df_dict = df_dict.append(dict(zip(keys_list_dict, values_list_dict)), ignore_index=True)

    # Export dfs
    df.to_csv(os.path.join(os.path.dirname(__file__), relative_path + 'p01_geu_cluster.csv'), index=False)
    df_dict.to_csv(os.path.join(os.path.dirname(__file__), relative_path + 'pbi_geus_info.csv'), index=False)

    if print_on: print('GEU info exported.')


def result_validation_data(path: str =
                           "C:/Users/facundo.scasso/Accenture/" +
                           "Telecom Inventory Optimization - Multi-Echelon - General/07. Resultados/Base/service.csv",
                           date: datetime = datetime(2019, 12, 23)):

    # Read csv and use last comparable date
    df = pd.read_csv(path, dtype={'geu': str}, date_parser=['fecha'])
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df[df['fecha'] == date].reset_index(drop=True)

    # Read pbi_geus_info
    relative_path = f'exportables/'
    df_dict = pd.read_csv(os.path.join(os.path.dirname(__file__), relative_path + 'pbi_geus_info.csv'))

    # Merge with df
    df = df.merge(df_dict, how='left')

    with pd.option_context('display.max_columns', None):
        print(df)

    return df


def download_geus(geus: dict, include_unigeus: bool = False, print_on: bool = True):
    """Download all geus as PNG files"""

    # Calculate geu count
    geu_count = 0
    if include_unigeus:
        geu_count = len(geus)
    else:
        for geu in geus.values():
            if not geu.is_unigeu:
                geu_count += 1

    if print_on:
        with tqdm(total=geu_count, file=sys.stdout) as pbar:
            for geu in geus.values():
                if include_unigeus:
                    geu.save_graph_to_png(extra_info=True)
                else:
                    if not geu.is_unigeu:
                        geu.save_graph_to_png(extra_info=True)

                # Manually update progress bar
                pbar.update(1)

    else:
        for geu in geus.values():
            if include_unigeus:
                geu.save_graph_to_png(extra_info=True)
            else:
                if not geu.is_unigeu:
                    geu.save_graph_to_png(extra_info=True)


def result_validation(df: pd.DataFrame, clf: bool = False, filter: dict = None, print_on=True,
                      locate_figures: bool = False, win_w: int = 1280, win_h: int = 1080,
                      plot: bool = True):
    # CALCULATIONS

    # Filter as requested (if applies)
    if filter:
        if print_on: print(f"Filter: {filter}")

        # Validation
        allowed_keys = ['geu', 'dominio', 'marca', 'criticidad', 'cluster', 'tipo abastecimiento', 'completo']
        for key in filter.keys():
            if key not in allowed_keys:
                raise KeyError('A filter column was not found. ' +
                               f'Please use (if any) one of the following: {allowed_keys}')

        # Filter
        for col, vals in filter.items():
            df = df[df[col].apply(lambda x: x in vals)]

        if print_on:
            print('Unique values left:')
            for col in filter.keys():
                print(col, df[col].unique())

    # Calculate service level
    df = df.groupby('heuristica').agg({'on_time_acumulado': 'sum',
                                       'total_acumulado': 'sum'})
    df['SL'] = df['on_time_acumulado'] / df['total_acumulado']
    df.sort_values('SL', inplace=True)

    # ---------------

    # PLOT

    if not plot:
        return df
    else:
        # Figure
        fig_name = '(RES) Service Level por Heuristica'
        if filter:
            fig_name += f" - Filtro: {filter}"
        if clf:
            for i in plt.get_fignums():
                if plt.figure(i).get_label()[0:5] == "(RES)":
                    plt.close(plt.figure(i).get_label())
        # Close plot with the same name as the one we're creating (if applies)
        for i in plt.get_fignums():
            if plt.figure(i).get_label() == fig_name:
                plt.close(fig_name)

        # Create a figure instance
        fig = plt.figure(fig_name)
        plt.suptitle('Service Level por Heurística', fontsize=18)
        if filter:
            plt.title(f'Filtrado por: {filter}', fontsize=12)

        # Put figure in the corresponding corner
        if locate_figures:
            locate_figure(win_w, win_h, where='full')

        # Create an axes instance
        ax = fig.add_subplot(111)

        ax.barh(y=df.index, width=df['SL'], edgecolor='black',
                color=['lightblue' if hist == 'historia' else acn_colors[0] for hist in list(df.index)])
        for i, v in enumerate(df['SL']):
            ax.text(v + 0.005, i, round(v, 4), va='center')

        # x axis
        plt.xlabel('Service Level')
        ax.set_xlim([0, 1])
        ticks = np.around(np.arange(0, 1.1, 0.1), 2)
        plt.xticks(ticks, ticks)

        plt.show()
