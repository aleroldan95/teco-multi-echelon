import pandas as pd
import datetime as dt

import networkx as nx


def check_geus_islands(geus_by_id):
    """
    Checks if there is a GEU that has two or more disconnected subgraphs (referred as 'islands')
    :param geus_by_id: GEU dict
    :return: Void
    """
    print("Checking GEUs islands...")

    geus_with_islands = []
    for geu in geus_by_id.values():
        graph_undir = geu.graph.to_undirected()
        islands = [graph_undir.subgraph(c).copy() for c in nx.connected_components(graph_undir)]
        if len(islands) > 1:
            geus_with_islands.append(geu.catalog)

    if len(geus_with_islands) == 0:
        print('There are no GEUs with islands')
    elif len(geus_with_islands) == 1:
        print('There is', 1, 'GEU with islands:', geus_with_islands)
    else:
        print('There are', len(geus_with_islands), 'geus with islands:', geus_with_islands)


def check_geus_uniqueness(geus_by_id):
    """
    Checks if every scope material is assigned to just 1 GEU
    The verification is done by iterating all nodes of the GEUs generated with scope material
    This is done so that we know for sure that there are no repetitions
    :param geus_by_id: GEU dict
    :return: Void
    """

    print("Checking GEUs uniqueness...")

    node_list = []
    repetitions = {}
    for geu in geus_by_id.values():
        for node in geu.graph:
            # If it is in the list of nodes, it's repeated
            if node in node_list:
                # If it is the first repetition, add the key
                if node not in repetitions.keys():
                    repetitions[node] = []
                # Add repetition
                repetitions[node].append(geu.catalog)
            # If it isn't in the list of nodes, add it
            else:
                node_list.append(node)

    # Print accordingly
    if len(repetitions) == 0:
        print("No scope material is in two different GEUs")
    elif len(repetitions) == 1:
        print('There is', 1, 'material that belongs to multiple GEUs:', repetitions)
    else:
        print('There are', len(repetitions), 'materials that belong to multiple GEUs:', repetitions)


def check_geus_strength(geus_by_id):
    """
    Checks for weakly connected GEUs
    This is done so that we know if there is any GEU for which we shouldn't assume
    that every material replaces each other
    :param geus_by_id: GEU dict
    :return: Void
    """

    print("Checking GEUs strength...")

    weakly_connected_geus = []
    # Checking for weakly connected GEUs
    for geu in geus_by_id.values():
        if geu.is_strongly_connected is False:
            weakly_connected_geus.append(geu.catalog)

    # Print accordingly
    if len(weakly_connected_geus) == 0:
        print('All GEUs are strongly connected')
    elif len(weakly_connected_geus) == 1:
        print('There is', 1, 'weakly connected GEUs:', weakly_connected_geus)
    else:
        print('There are', len(weakly_connected_geus), 'weakly connected GEUs:', weakly_connected_geus)


def check_geus(geus_by_id):
    """
    Does in 1 statement all check_geus functions
    :param geus_by_id: GEU dict
    :return: Void
    """
    check_geus_uniqueness(geus_by_id)
    check_geus_islands(geus_by_id)
    check_geus_strength(geus_by_id)
    print()
    print('Finished checking GEUs.')


def export_geus_info(geus_by_id: {}, date: dt.datetime, print_on: bool = True):
    # Generating df data
    data = []
    for geu in geus_by_id.values():
        # Making a string that can be used in the df
        catalog_string = ''
        materials_in_geu = len(geu.materials)
        counter = 1
        for mat in geu.materials:
            catalog_string += mat.catalog
            if counter < materials_in_geu:
                catalog_string += ', '
                counter += 1

        # Append to data
        data.append([geu.catalog, geu.domain, geu.name, len(geu.materials), catalog_string, geu.is_strongly_connected,
                     geu.criticality, str(geu.leadtime) + ' +/- ' + str(geu.leadtime_sd)])

    # Making df
    df_export = pd.DataFrame(data, columns=['GEU', 'Dominio', 'Descripción', 'Conteo mat.', 'Materiales', 'Completo',
                                            'Criticidad', 'Lead-time'])

    file = f'geus_info[{date.date()}].xlsx'
    df_export.to_excel('Outputs/GEUs/' + file)
