import numpy as np
import pandas as pd
import itertools
from math import pi
import streamlit as st
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler
from pylab import rcParams

from typing import Union

from classes.material import Material
from classes.geu import GEU


def validation_data(group: Union[GEU, Material]):
    both_vars = ['Catalog', 'Price', 'Price Type', 'Buyable',
                 'Criticality', 'Has Criticality Data', 'Domain']
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


def validation(df, clusters: int=5):
    # Color palette
    acn_colors = ['#a100ff', '#7600c0', '#460073',
                  '#00baff', '#008eff', '#004dff',
                  '#969696', '#5a5a5a', '#000000']

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return '{v:d} ({p:.0f}%)'.format(p=pct, v=val)

        return my_autopct

    def draw_pie(figure: int, series: pd.Series, variable: str, sorted_colors: bool = False,
                 where: str = 'left', cdict: dict = None, text_color: str = 'white'):
        # Figure
        title = str(figure) + f' Pie {variable}'
        figures[title] = plt.figure(title)
        plt.title(f'Distribución de {variable}', fontsize=18)

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

    def draw_histogram(figure: int, series: pd.Series, variable: str, where: str = 'left'):
        # Figure
        title = str(figure) + f' Hist. {variable}'
        figures[title] = plt.figure(title)

        # Subplot 1: Histogram as-is
        plt.subplot(211)
        plt.title(f'Distribución de {variable} en USD', fontsize=18)

        # Subplot 1: Histogram as-is
        plt.hist(series, edgecolor='black', linewidth=0.8)

        # Axis
        plt.gca().set(ylabel='Frecuencia')
        fmt = '${x:,.0f}'
        tick = mtick.StrMethodFormatter(fmt)
        plt.gca().xaxis.set_major_formatter(tick)

        # Subplot 2: Zoomed for range 0-3000 USD
        plt.subplot(212)
        plt.hist(series, edgecolor='black', linewidth=0.8, range=[0, 3000])

        # Axis
        plt.gca().set(xlabel=f'{variable} [US$]',
                      ylabel='Frecuencia')
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
        title = str(figure) + ' Radar clusters'
        figures[title] = plt.figure(title)
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


    def draw_bubbleplot(figure: int, df: pd.DataFrame, raw: bool = False, where: str = 'left'):
        # Create a figure instance
        if raw:
            title = str(figure) + 'Tipo Abastecimiento (Raw)'
            figures[title] = plt.figure(title)
            plt.title('Tipo de Abastecimiento (Raw)', fontsize=18)
        else:
            title = str(figure) + 'Tipo Abastecimiento (Crudo)'
            figures[title] = plt.figure(title)
            plt.title('Tipo de Abastecimiento', fontsize=18)

        # Create an axes instance
        ax = figures[title].add_subplot(111)

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
        title = f'(VAL) {figure} Leadtime Tipo Abastecimiento'
        figures[title] = plt.figure(title)
        plt.title('Leadtime por Tipo de Abastecimiento', fontsize=18)

        # Make violinplot
        sns.violinplot(x=x, y=y)

    # Initialize fig_num
    fig_num = 1

    # ----------------------------------------------------
    figures = {}

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
                 cdict=colors, text_color='white')
        fig_num += 1
    if 'Raw Criticality' in df.columns:
        draw_pie(figure=fig_num, series=df['Raw Criticality'],
                 variable='Información de Criticidad', where='right',
                 cdict={'bajo': acn_colors[0], 'mayor': acn_colors[1], 'critico': acn_colors[2]})
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

    # Format data input
    features = {
                'Weighted Demand v2 (TKT)': ['Demanda armónica', 'u/período de clust.'],
                'WH with Demand (TKT)': ['Cantidad de GREPs con demanda', 'Total GREPs'],
                'Last Ticket ReLu 180': ['Días desde último ticket ReLu 180', 'días'],
                'Average Peak (TKT)': ['Pico promedio de demanda', 'u'],
                'Highest Peak (TKT)': ['Pico máximo de demanda', 'u']
                }
    # Clustering

    # Pie chart
    if 'Cluster' in df.columns:
        draw_pie(figure=fig_num, series=df['Cluster'], variable='Cluster', sorted_colors=True, where='left')
        fig_num += 1

    return figures

