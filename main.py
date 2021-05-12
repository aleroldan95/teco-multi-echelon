from data_loader_class import DataClass
from functions import main_timer
import pyfiglet
from validation import *
import datetime as dt


def main_process():

    # Start main timer
    tic = main_timer(True)

    # Figlet text
    custom_fig = pyfiglet.Figlet(font='slant')
    print("\n\nWelcome to")
    print(custom_fig.renderText('Neural Teco'))

    print_on = True
    data_class = DataClass(print_on=print_on, is_streamlit=False, bar_progress=None,
                           grep_type='greps', demand_type='tickets', start_date=dt.datetime(2020, 1, 2), is_s4=True)
    materials_by_id, geus_by_id, wh_list, conn, k_clusters = data_class.return_values()

    # ------------------------------------------------------------------------------------------------------------------
    # Stop Main timer
    main_timer(False, tic, is_streamlit=False)

    return materials_by_id, geus_by_id, wh_list, conn, k_clusters, data_class


if __name__ == "__main__":
    materials_by_id, geus_by_id, wh_list, conn, k_clusters, data_class = main_process()
