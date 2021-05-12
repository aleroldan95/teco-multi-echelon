import datetime as dt
from data_loader_class import DataClass
from functions import to_datetime


class StockInTransit:

    def __init__(self, receiving_wh: str, date: dt.date, amount: int, leadtime: int = None, donating_wh: str = None):

        self.amount = amount

        self.receiving_wh = receiving_wh
        self.donating_wh = donating_wh
        if self.donating_wh is None:  # It's a purchase movement
            self.is_purchase = True
            self.is_redistribution = False
            self.donating_wh = 'Buenos Aires'  # All purchases arrive to 'Buenos Aires'
        else:
            self.is_redistribution = True
            self.is_purchase = False

        self.leadtime = leadtime
        self.emission_date = date
        # self.arrival_date = self.emission_date + dt.timedelta(days=round(self.leadtime))
        self.arrival_date = to_datetime(self.emission_date + dt.timedelta(days=self.leadtime))

    def __repr__(self):
        return "<Receiving_wh>: {} <donating_wh>: {} <Cantidad>: {} <Arrival_date>: {}".format(
            self.receiving_wh, self.donating_wh, self.amount, self.arrival_date)
