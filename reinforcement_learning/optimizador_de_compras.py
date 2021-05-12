import os
import datetime as dt
from reinforcement_learning.model_vs_history import ModelVSHistory
from data_loader_class import DataClass
import pickle as pkl
import ray
import streamlit as st

def optimizar(sl_optimo, resupplies_per_year, cluster=None, domain=None, chosen_heuristic="Action1", step=(.1, 1), is_streamlit=False,
              grep_type='7_crams', start_date=dt.datetime(2018, 7, 1), end_date=dt.datetime(2020, 1, 1)):
    if cluster:
        clusters = [cluster]
    else:
        clusters = DataClass.clusters
    if domain:
        domains = [domain]
    else:
        domains = list(DataClass.load_from_file(grep_type='7_crams').domain_list)

    multiplicadores = {dom: [(0.5, 0)] * len(DataClass.clusters) for dom in domains}
    env_config = {"start_date": start_date, "end_date": end_date, "get_info": True,
                  'grep_type': grep_type, "from_pickle": False, 'use_historic_stocks': False,
                  'resupplies_per_year': resupplies_per_year, "multiplicador": multiplicadores}

    minimos_representativos = {'ACCESO - FIJA': [100, 40, 20], 'ACCESO - MOVIL': [100, 40, 28], 'CORE VOZ': [100, 40, 20],
                               'ENTORNO': [100, 40, 20], 'TRANSPORTE - DX': [100, 40, 40], 'TRANSPORTE - TX': [100, 40, 20],
                               'TX - RADIOENLACES Y SATELITAL': [100, 20, 20], # y just in case:
                               'CMTS': [100, 40, 20], 'Many': [100, 20, 20]}


    model_vs_history = ModelVSHistory(env_config, chosen_heuristics=[chosen_heuristic])
    for dom in domains:
        for clu in clusters:
            model_vs_history.reset_data()
            cant = len(list(filter(lambda x: x.cluster == clu and x.domain == dom, model_vs_history.env.data_class.geus_by_id.values())))
            if cant == 0:
                continue
            rep = min(cant, minimos_representativos[dom][clu - 1])
            model_vs_history.run(slice=rep, clusters=[clu], domains=[dom])
            message = f'**Dominio:** {dom}  **Cluster:** {clu}  **Compras:** {multiplicadores[dom][clu - 1][0]*100}%  **SS:** {multiplicadores[dom][clu - 1][1]} u  **Disponibilidad de Repuesto:** {model_vs_history.return_service_level()} %'
            if is_streamlit: st.write(message)
            else: print(message)

            while model_vs_history.return_service_level() < sl_optimo*100:
                model_vs_history.reset_data()

                if clu in [1, 2]:
                    multiplicadores[dom][clu - 1] = (1, multiplicadores[dom][clu - 1][1] + step[1])
                else:
                    if multiplicadores[dom][clu - 1][0] < 1:
                        multiplicadores[dom][clu - 1] = (multiplicadores[dom][clu - 1][0] + step[0], 0)
                    else:
                        multiplicadores[dom][clu - 1] = (1, multiplicadores[dom][clu - 1][1] + step[1])

                model_vs_history.set_multipliers(multiplicadores)
                model_vs_history.run(slice=rep, clusters=[clu], domains=[dom])

                message = f'**Dominio:** {dom}  **Cluster:** {clu}  **Compras:** {multiplicadores[dom][clu - 1][0]*100}%  **SS:** {multiplicadores[dom][clu - 1][1]} u  **Disponibilidad de Repuesto:** {model_vs_history.return_service_level()} %'
                if is_streamlit: st.markdown(message)
                else:print(message)

            if is_streamlit:
                message = f'**Compras:** {multiplicadores[dom][clu - 1][0]*100}%  **Stock de Seguridad:** {multiplicadores[dom][clu - 1][1]} u  **Disponibilidad de Repuesto Alcanzado:** {model_vs_history.return_service_level()} %'
                st.markdown(message)
                st.write("---")
            else:print("Siguiente")
    if chosen_heuristic == "Neural Net":
        ray.shutdown()
    return multiplicadores


def save_multipliers(multiplicadores, resupplies_per_year, grep_type, cluster=None, is_streamlit=False):
    if is_streamlit:
        with open(os.path.join(os.path.dirname(__file__), '../pickles/multiplicadores.pkl'), "rb") as file:
            saved_multi = pkl.load(file)
    else:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{grep_type}_{resupplies_per_year}.pkl'), "rb") as file:
            saved_multi = pkl.load(file)
    if cluster:
        clusters = [cluster]
    else:
        clusters = DataClass.clusters
    for domain in multiplicadores.keys():
        if domain in saved_multi.keys():
            initial = saved_multi[domain]
        else:
            initial = [(1, 0)] * len(DataClass.clusters)
        for clu in clusters:
            initial[clu - 1] = multiplicadores[domain][clu - 1]
        saved_multi[domain] = initial
    if is_streamlit:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores.pkl'),
                  "wb") as file:
            pkl.dump(saved_multi, file)
    else:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{grep_type}_{resupplies_per_year}.pkl'), "wb") as file:
            pkl.dump(saved_multi, file)


def reset_multipliers(resupplies_per_year, grep_type, is_streamlit = False):

    multiplicadores = {dom: [(1, 0)] * len(DataClass.clusters) for dom in DataClass.load_from_file().domain_list}

    if is_streamlit:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores.pkl'),
                  "wb") as file:
            pkl.dump(multiplicadores, file)
    else:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{grep_type}_{resupplies_per_year}.pkl'), "wb") as file:
            pkl.dump(multiplicadores, file)


def get_multipliers( resupplies_per_year, grep_type, is_streamlit=False):
    if is_streamlit:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores.pkl'),
                  "rb") as file:
            multi = pkl.load(file)
    else:
        with open(os.path.join(os.path.dirname(__file__), f'../pickles/multiplicadores_{grep_type}_{resupplies_per_year}.pkl'), "rb") as file:
            multi = pkl.load(file)
    return multi


if __name__ == "__main__":

    grep_type = "7_crams"
    #
    m = optimizar(0.35, 1, grep_type=grep_type, chosen_heuristic='Action1', cluster=1)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=1, cluster=1)
    m = optimizar(0.35, 1, grep_type=grep_type, chosen_heuristic='Neural Net', cluster=2)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=1, cluster=2)
    m = optimizar(0.3, 1, grep_type=grep_type, chosen_heuristic='Neural Net', cluster=3)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=1, cluster=3)

    m = optimizar(0.35, 4, grep_type=grep_type, chosen_heuristic='Action1', cluster=1)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=4, cluster=1)
    m = optimizar(0.35, 4, grep_type=grep_type, chosen_heuristic='Neural Net', cluster=2)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=4, cluster=2)
    m = optimizar(0.3, 4, grep_type=grep_type, chosen_heuristic='Neural Net', cluster=3)
    save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=4, cluster=3)

    print(get_multipliers(is_streamlit=False, grep_type="7_crams", resupplies_per_year=1))
    # print(get_multipliers(is_streamlit=False, grep_type=grep_type, resupplies_per_year=1))
    print(get_multipliers(is_streamlit=False, grep_type="7_crams", resupplies_per_year=4))
    # print(get_multipliers(is_streamlit=False, grep_type=grep_type, resupplies_per_year=4))

    # m = {key: [value[0], (value[1][0], value[1][1] * 2), (1, 0)] for key, value in get_multipliers(is_streamlit=False, grep_type="7_crams", resupplies_per_year=1).items()}
    # save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=1, cluster=3)
    # m = {key: [value[0], (value[1][0], value[1][1] * 2), (1, 0)] for key, value in
    #      get_multipliers(is_streamlit=False, grep_type="7_crams", resupplies_per_year=4).items()}
    # save_multipliers(multiplicadores=m, grep_type=grep_type, resupplies_per_year=4, cluster=3)
