import pandas as pd
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
import warnings
warnings.simplefilter("ignore")
import numpy as np

def load_data():
    # Function that loads input dataframe
    df = pd.read_csv("exportables/monthly_demand.csv")

    # Create a dataframe that has info of month, quarter and semester
    df_dates = pd.DataFrame(data={
        'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'month_str': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
        'semester': ['01', '01', '01', '01', '01', '01', '02', '02', '02', '02', '02', '02'],
        'quarter': ['01', '01', '01', '02', '02', '02', '03', '03', '03', '04', '04', '04'],
    })
    return df, df_dates


def get_min_date_per_geu(df):
    # Create column
    df_sales = df.copy()
    df_sales = df_sales[df_sales['demand'] > 0].reset_index(drop=True)

    # Get minimum date
    df_min = df_sales[['geu', 'period']].groupby(['geu']).min().reset_index()

    df_min.rename(columns={'period': 'min_date'}, inplace=True)

    min_date_dict = {}
    for index, row in df_min.iterrows():
        min_date_dict[row['geu']] = row['min_date']

    return min_date_dict


def expand_df(df):
    # As SKUs are not having all months of data, we

    # Dataframe with all months
    df_months = df[['year', 'month']].drop_duplicates().reset_index(drop=True)
    df_months['key'] = 0

    # Dataframe with all geus
    df_geus = df[['geu']].drop_duplicates().reset_index(drop=True)
    df_geus['key'] = 0

    # Cross join
    df_months_geus = df_months.merge(df_geus, how='outer')

    # Drop key column
    df_months_geus.drop(columns=['key'], inplace=True)

    # Merge with original df
    df = pd.merge(df_months_geus, df, how='left', on=['year', 'month', 'geu'])
    df['demand'] = df['demand'].fillna(0)

    return df


def group_by_period(df, df_dates, period):
    # Function that will group input information by a particular period (semester, or quarter)

    # Filter by period
    df_dates = df_dates[['month', period]]

    # Merge information
    df = pd.merge(df, df_dates, how='left', on=['month'])

    # Create Year-Period column
    df[f'year-{period}'] = df['year'].astype(str) + df[period]
    df[f'year-{period}'] = df[f'year-{period}'].astype(int)

    # Group by GEU and year period
    df = df[[f'year-{period}', 'geu', 'demand']].groupby([f'year-{period}', 'geu']).sum().reset_index()

    return df


def naive_forecast(df, min_date_dict, period):
    print("Producing naive forecast")
    # Run a naive forecast

    # List of available periods
    periods = df['period'].unique()

    # Run forecast for all weeks
    df_naive = pd.DataFrame()

    for last_period in periods:
        df_naive_week = run_naive_forecast_week(df, min_date_dict, last_period, period)
        df_naive = pd.concat([df_naive, df_naive_week], ignore_index=True)

    return df_naive


def run_naive_forecast_week(df, min_date_dict, last_period_available, period):
    # Function that creates a naive Forecast for all GEUs

    # List of GEUs
    geus = df['geu'].unique()

    # We check which period we are going to forecast
    if period == 'month_str':
        last_period_year = '12'
    elif period == 'semester':
        last_period_year = '02'
    else:
        last_period_year = '04'

    # If the last available period is the last period of an existing year, we change the year
    last_period_string = str(last_period_available)
    if last_period_string[-2:] == last_period_year:
        fcst_yr = last_period_string[:4] + "01"
        forecasted_period = int(fcst_yr)
        forecasted_period += 100
    else:
        forecasted_period = last_period_available + 1

    # Create output sku
    df_naive_week = pd.DataFrame()

    for geu in geus:
        # Filter result
        df_geu = df[(df['geu'] == geu) & (df['period'] == last_period_available)].reset_index(drop=True)

        # If this GEU had a previous sale, we add it to the output dataframe
        if last_period_available >= min_date_dict[geu]:
            df_geu['period'] = forecasted_period
            df_geu.rename(columns={'demand': 'naive_forecast'}, inplace=True)
            df_naive_week = pd.concat([df_naive_week, df_geu], ignore_index=True)

    return df_naive_week


def run_expo_forecast_week(df, min_date_dict, last_period_available, period, alpha):
    # Function that creates a naive Forecast for all GEUs

    # List of GEUs
    geus = df['geu'].unique()

    # We check which period we are going to forecast
    if period == 'month_str':
        last_period_year = '12'
    elif period == 'semester':
        last_period_year = '02'
    else:
        last_period_year = '04'

    # If the last available period is the last period of an existing year, we change the year
    last_period_string = str(last_period_available)
    if last_period_string[-2:] == last_period_year:
        fcst_yr = last_period_string[:4] + "01"
        forecasted_period = int(fcst_yr)
        forecasted_period += 100
    else:
        forecasted_period = last_period_available + 1

    # Create output sku
    df_expo_week = pd.DataFrame()

    for geu in geus:
        # Filter result
        df_geu = df[(df['geu'] == geu) & (df['period'] <= last_period_available)].reset_index(drop=True)

        # If this GEU had a previous sale and there is more than two weeks of data, we add it to the output dataframe
        if last_period_available > min_date_dict[geu] and len(df_geu) >= 2:
            # Create and fit model
            model = SimpleExpSmoothing(df_geu['demand'].values).fit(
                optimized=False, smoothing_level=alpha)

            # Create output df
            df_forecast_geu = pd.DataFrame({
                'geu': [geu],
                'period': [forecasted_period],
                f'expo_{alpha}_forecast': list(model.forecast(1))
            })
            df_expo_week = pd.concat([df_expo_week, df_forecast_geu], ignore_index=True)

    return df_expo_week


def expo_forecast(df, min_date_dict, period, alpha):
    print(f"Producing expo forecast with alpha {alpha}")
    # Run a naive forecast

    # List of available periods
    periods = df['period'].unique()

    # Run forecast for all weeks
    df_expo = pd.DataFrame()

    for last_period in periods:
        df_expo_week = run_expo_forecast_week(df, min_date_dict, last_period, period, alpha)
        df_expo = pd.concat([df_expo, df_expo_week], ignore_index=True)

    return df_expo

def croston(data, alpha=0.3, target_column='target', date_column='week'):
    '''Implements an croston smoothing algorithm to generate forecast.
    The last week is a test prediction all the other ones are training.
    The function adds a column on the data frame that it was called upon and returns it.

    data: Has to be a data frame with a target_column and date_column
    target_column: Column name on the data data frame with target values
    date_column: Column name on the data data frame with target values'''
    try:
        # sets data
        data = data.sort_values(by=date_column, ascending=True)
        x = data[target_column]
        d = data[date_column]
        # sets variables
        N = x.size
        f = np.zeros((N,))
        y = np.zeros((N,))
        l = np.zeros((N,))
        # initialize
        f[0] = x[0]
        l[0] = x[0]
        y[0] = 1
        q = 1
        # forecast
        for i in range(1, N):
            # ex-post
            if d[i] < d.max():
                if x[i] <= 0:
                    l[i] = l[i - 1]
                    y[i] = y[i - 1]
                    q += 1
                else:
                    l[i] = (1 - alpha) * l[i - 1] + alpha * x[i]
                    y[i] = (1 - alpha) * y[i - 1] + alpha * q
                    q = 1
                f[i] = l[i] / y[i]
            # ex-ante
            else:
                f[i] = f[i - 1]
        # set name
        colName = 'Croston {}'.format(alpha)
        # add to data
        data[colName] = f
        return data
    except:
        print('Error in croston')


def run_croston_forecast_week(df, min_date_dict, last_period_available, period, alpha):
    # Function that creates a naive Forecast for all GEUs

    # List of GEUs
    geus = df['geu'].unique()

    # We check which period we are going to forecast
    if period == 'month_str':
        last_period_year = '12'
    elif period == 'semester':
        last_period_year = '02'
    else:
        last_period_year = '04'

    # If the last available period is the last period of an existing year, we change the year
    last_period_string = str(last_period_available)
    if last_period_string[-2:] == last_period_year:
        fcst_yr = last_period_string[:4] + "01"
        forecasted_period = int(fcst_yr)
        forecasted_period += 100
    else:
        forecasted_period = last_period_available + 1

    # Create output dataframe
    df_croston_week = pd.DataFrame()

    for geu in geus:
        # Filter result
        df_geu = df[(df['geu'] == geu) & (df['period'] <= last_period_available)].reset_index(drop=True)

        # If this GEU had a previous sale and there is more than two weeks of data, we add it to the output dataframe
        if last_period_available > min_date_dict[geu] and len(df_geu) >= 2:

            # Create an auxiliary df that will be appended to the original one
            df_aux = pd.DataFrame(data={
                'period': [forecasted_period],
                'geu': [geu],
                'demand': [None]
            })
            df_geu = pd.concat([df_geu, df_aux], ignore_index=True)

            # Run croston
            df_geu = croston(df_geu, alpha=alpha, target_column='demand', date_column='period')
            df_geu = df_geu[['period', 'geu', f'Croston {alpha}']]

            # Filter the week we want to keep and append it to output df
            df_forecast_geu = df_geu[df_geu['period'] == forecasted_period].reset_index(drop=True)

            df_croston_week = pd.concat([df_croston_week, df_forecast_geu], ignore_index=True)

    return df_croston_week


def croston_forecast(df, min_date_dict, period, alpha):
    print(f"Producing croston forecast with alpha {alpha}")
    # Run a naive forecast

    # List of available periods
    periods = df['period'].unique()

    # Run forecast for all weeks
    df_croston = pd.DataFrame()

    for last_period in periods:
        df_croston_week = run_croston_forecast_week(df, min_date_dict, last_period, period, alpha)
        df_croston = pd.concat([df_croston, df_croston_week], ignore_index=True)

    return df_croston


def run_linear_forecast_week(df, min_date_dict, last_period_available, period):
    # Function that creates a naive Forecast for all GEUs

    # List of GEUs
    geus = df['geu'].unique()

    # We check which period we are going to forecast
    if period == 'month_str':
        last_period_year = '12'
    elif period == 'semester':
        last_period_year = '02'
    else:
        last_period_year = '04'

    # If the last available period is the last period of an existing year, we change the year
    last_period_string = str(last_period_available)
    if last_period_string[-2:] == last_period_year:
        fcst_yr = last_period_string[:4] + "01"
        forecasted_period = int(fcst_yr)
        forecasted_period += 100
    else:
        forecasted_period = last_period_available + 1

    # Create output dataframe
    df_linear_week = pd.DataFrame()

    for geu in geus:
        # Filter result
        df_geu = df[(df['geu'] == geu) & (df['period'] <= last_period_available)].reset_index(drop=True)

        # If this GEU had a previous sale and there is more than two weeks of data, we add it to the output dataframe
        if last_period_available > min_date_dict[geu] and len(df_geu) >= 2:
            # Create and fit model
            model = LinearRegression()
            model.fit(np.array(range(df_geu.shape[0])).reshape(-1, 1), df_geu['demand'].values.reshape(-1, 1))

            # Create output df
            df_forecast_geu = pd.DataFrame({
                'geu': [geu],
                'period': [forecasted_period],
                'Linear': model.predict([[df_geu.shape[0]]])[0]
            })
            df_linear_week = pd.concat([df_linear_week, df_forecast_geu], ignore_index=True)

    return df_linear_week


def linear_forecast(df, min_date_dict, period):
    print(f"Producing linear forecast")
    # Run a naive forecast

    # List of available periods
    periods = df['period'].unique()

    # Run forecast for all weeks
    df_linear = pd.DataFrame()

    for last_period in periods:
        df_linear_week = run_linear_forecast_week(df, min_date_dict, last_period, period)
        df_linear = pd.concat([df_linear, df_linear_week], ignore_index=True)

    return df_linear


def main():
    # Load data
    df, df_dates = load_data()

    # As original data only has weeks with sales, we add additional weeks
    df = expand_df(df)

    # Group by period
    period = 'quarter'
    df = group_by_period(df, df_dates, period)
    df.rename(columns={f'year-{period}': 'period'}, inplace=True)

    # Calculate minimum week of sales per GEU
    min_date_dict = get_min_date_per_geu(df)

    # Run naive forecast
    df_naive = naive_forecast(df, min_date_dict, period)

    # Run linear model
    df_linear = linear_forecast(df, min_date_dict, period)

    # Run expo forecasts
    df_expo_03 = expo_forecast(df, min_date_dict, period, 0.3)
    df_expo_05 = expo_forecast(df, min_date_dict, period, 0.5)
    df_expo_07 = expo_forecast(df, min_date_dict, period, 0.7)

    # Run croston forecasts
    df_croston_03 = croston_forecast(df, min_date_dict, period, 0.3)
    df_croston_05 = croston_forecast(df, min_date_dict, period, 0.5)
    df_croston_07 = croston_forecast(df, min_date_dict, period, 0.7)

    # Final output construction
    df_output = pd.merge(df, df_naive, how='inner', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_linear, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_expo_03, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_expo_05, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_expo_07, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_croston_03, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_croston_05, how='left', on=['period', 'geu'])
    df_output = pd.merge(df_output, df_croston_07, how='left', on=['period', 'geu'])

    # Fill NAs with Naive
    df_output['Linear'] = df_output['Linear'].fillna(df_output['naive_forecast'])
    df_output['expo_0.3_forecast'] = df_output['expo_0.3_forecast'].fillna(df_output['naive_forecast'])
    df_output['expo_0.5_forecast'] = df_output['expo_0.5_forecast'].fillna(df_output['naive_forecast'])
    df_output['expo_0.7_forecast'] = df_output['expo_0.7_forecast'].fillna(df_output['naive_forecast'])
    df_output['Croston 0.3'] = df_output['Croston 0.3'].fillna(df_output['naive_forecast'])
    df_output['Croston 0.5'] = df_output['Croston 0.5'].fillna(df_output['naive_forecast'])
    df_output['Croston 0.7'] = df_output['Croston 0.7'].fillna(df_output['naive_forecast'])

    model_cols = ['Naive', 'Linear', 'expo_0.3_forecast', 'expo_0.5_forecast', 'expo_0.7_forecast', 'Croston 0.3',
                  'Croston 0.5', 'Croston 0.7']
    for col in model_cols:
        df_output[col+'_error'] = df_output[col] - df_output['demand']

    maes = pd.DataFrame(df_output[model_cols].abs().sum(axis=0)/df_output.shape[0], columns=['mae'])

    # Save results
    maes.to_csv("exportables/forecast_maes.csv")

main()