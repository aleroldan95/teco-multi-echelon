import pandas as pd


class Warehouse:

    # Static Attributes

    distances_df = pd.DataFrame()

    def __init__(self, address: str, lat: float, long: float):

        # Location
        self.address = address
        self.lat = lat
        self.long = long
