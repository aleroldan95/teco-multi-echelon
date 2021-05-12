import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from numpy.core.defchararray import upper


class Period:

    # Constructors------------------------------------------------------------------------------------------------------

    def __init__(self, date, unit: str, units_in_period: int, date_is_first_date=True):
        # Sets date to have no time part
        self.first_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        self.units_in_period = units_in_period
        self.unit = str(upper(unit))
        self.date_is_first_date = date_is_first_date
        self.__set_last_date()
    
    # ==================================================================================================================
    
    def __repr__(self):
        return f"<Period> date:{self.first_date.date()}, units: {self.units_in_period} {self.unit}"
    
    # Setters-----------------------------------------------------------------------------------------------------------
    
    def __set_last_date(self):
        
        # Days
        if self.unit == "D":
            self.last_date = self.first_date + relativedelta(days=self.units_in_period) - relativedelta(days=1)
            self.unit_name_in_spanish = "día"
            if self.units_in_period > 1:
                self.unit_name_in_spanish += "s"
        
        # Weeks
        elif self.unit == "W":
            # Period in Weeks - Starts on Monday of the given week
            # Note: Weekday function returns 0 for Monday
            if not self.date_is_first_date:
                self.first_date = self.first_date - relativedelta(days=self.first_date.weekday())
            self.last_date = self.first_date + relativedelta(weeks=self.units_in_period) - relativedelta(days=1)
            self.unit_name_in_spanish = "semana"
            if self.units_in_period > 1:
                self.unit_name_in_spanish += "s"
        
        # Months
        elif self.unit == "M":
            # Period in Months - Starts on the first day of the given month
            if not self.date_is_first_date:
                self.first_date = dt.datetime(self.first_date.year, self.first_date.month, 1)
            self.last_date = self.first_date + relativedelta(months=self.units_in_period) - relativedelta(days=1)
            self.unit_name_in_spanish = "mes"
            if self.units_in_period > 1:
                self.unit_name_in_spanish += "es"
        
        # Years
        elif self.unit == "Y":
            # Period in Years - Starts on January 1st of the given year
            if not self.date_is_first_date:
                self.first_date = dt.datetime(self.first_date.year, 1, 1)
            self.last_date = self.first_date + relativedelta(years=self.units_in_period) - relativedelta(days=1)
            self.unit_name_in_spanish = "año"
            if self.units_in_period > 1:
                self.unit_name_in_spanish += "s"
        
        else:
            print("Error - The Period class takes the following values for the unit argument:")
            print("D - Days")
            print("W - Weeks")
            print("M - Months")
            print("Y - Years")

    # Getters-----------------------------------------------------------------------------------------------------------

    def next(self):
        # Returns next period (in the specified units)
        return Period(self.last_date + relativedelta(days=1), self.unit, self.units_in_period, self.date_is_first_date)

    def previous(self):
        # Returns previous period (in the specified units)
        if self.unit == "D":
            return Period(self.first_date -
                          relativedelta(days=self.units_in_period), "D", self.units_in_period, self.date_is_first_date)
        elif self.unit == "W":
            return Period(self.first_date -
                          relativedelta(weeks=self.units_in_period), "W", self.units_in_period, self.date_is_first_date)
        elif self.unit == "M":
            return Period(self.first_date -
                          relativedelta(months=self.units_in_period), "M", self.units_in_period, self.date_is_first_date)
        elif self.unit == "Y":
            return Period(self.first_date -
                          relativedelta(years=self.units_in_period), "Y", self.units_in_period, self.date_is_first_date)

    def contains(self, date):
        return self.first_date <= date < self.last_date + relativedelta(days=1)

    def last_year_period(self):
        # Returns a Period Object of the same caracteristics (units, units in period) but starting an year prior
        return Period(self.first_date - relativedelta(years=1), self.unit, self.units_in_period)

    def next_year_period(self):
        # Returns a Period Object of the same caracteristics (units, units in period) but starting an year later
        return Period(self.first_date + relativedelta(years=1), self.unit, self.units_in_period)

    def days(self):
        return pd.date_range(self.first_date, self.last_date)

    def get_units_in_spanish(self, sep="_"):
        message = str(self.units_in_period) + sep + self.unit_name_in_spanish
        return message

    def days_elapsed(self):
        delta = self.last_date - self.first_date
        return delta.days + 1
