import alpaca_trade_api as tradeapi
import config
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta



api = tradeapi.REST(config.PAPER_API_KEY, config.PAPER_SECRET_KEY, config.PAPER_BASE_URL, api_version='v2')


def prepost_train_test_validate_offset_data(api, ticker, interval, train_days=180, test_days=60, validate_days=30,
                                            offset_days=0):
    ticker_data_dict = None
    ticker_data_dict = {}
    monthly_data_dict = None
    monthly_data_dict = {}
    interval_loop_data = None
    interval_loop_data = pd.DataFrame()
    stock_data = None

    days_to_collect = train_days + test_days + validate_days + offset_days

    TZ = 'US/Eastern'

    start = pd.to_datetime((datetime.now() - timedelta(days=days_to_collect)).strftime("%Y-%m-%d %H:%M"), utc=True)
    end = pd.to_datetime(datetime.now().strftime("%Y-%m-%d %H:%M"), utc=True)

    stock_data = api.get_bars(ticker, interval, start = start.isoformat(), end=end.isoformat(), adjustment="raw").df

    interval_loop_data = interval_loop_data.append(stock_data)
    df_start_ref = interval_loop_data.index[0]
    start_str_ref = pd.to_datetime(start, utc=True)

    while start_str_ref.value < ( pd.to_datetime(df_start_ref, utc=True) - pd.Timedelta(days=2.5)).value:
        end_new = pd.to_datetime(interval_loop_data.index[0].strftime("%Y-%m-%d %H:%M"), utc=True).isoformat()
        stock_data_new = None
        stock_data_new = api.get_bars(ticker, interval, start=start, end=end_new, adjustment="raw").df
        #stock_data_new = stock_data_new.reset_index()
        interval_loop_data = interval_loop_data.append(stock_data_new).sort_values(by=['index'], ascending=True)
        df_start_ref = interval_loop_data.index[0]

    stock_yr_min_df = interval_loop_data.copy()
    stock_yr_min_df["Open"] = stock_yr_min_df['open']
    stock_yr_min_df["High"]= stock_yr_min_df["high"]
    stock_yr_min_df["Low"] = stock_yr_min_df["low"]
    stock_yr_min_df["Close"] = stock_yr_min_df["close"]
    stock_yr_min_df["Volume"] = stock_yr_min_df["volume"]
    stock_yr_min_df["VolumeWeightedAvgPrice"] = stock_yr_min_df["vwap"]
    stock_yr_min_df["Time"] = stock_yr_min_df.index.tz_convert(TZ)
    stock_yr_min_df.index = stock_yr_min_df.index.tz_convert(TZ)
    final_df = stock_yr_min_df.filter(["Time", "Open", "High", "Low", "Close", "Volume", "VolumeWeightedAvgPrice"], axis = 1)

    first_day = final_df.index[0]
    traintest_day = final_df.index[-1] - pd.Timedelta(days= test_days+validate_days+offset_days)
    valtest_day = final_df.index[-1] - pd.Timedelta(days= test_days+offset_days)
    last_day = final_df.index[-1] - pd.Timedelta(days= offset_days)
    training_df =  final_df.loc[first_day:traintest_day] #(data_split - pd.Timedelta(days=1))]
    validate_df = final_df.loc[traintest_day:valtest_day]
    testing_df =  final_df.loc[valtest_day:last_day]
    full_train = final_df.loc[first_day:last_day]
    offset_df =  final_df.loc[last_day:]

    return training_df, validate_df, testing_df, full_train, offset_df, final_df, traintest_day, valtest_day


def timeFilterAndBackfill(df):
    """
    Prep df to be filled out for each trading day:
    Time Frame: 0730-1730
    Backfilling NaNs
    Adjusting Volume to Zero if no Trading data is present
       - Assumption is that there were no trades duing that time
    """

    df = df.between_time('07:29','17:26')

    TZ = 'US/Eastern'

    start_dateTime = pd.Timestamp(year = df.index[0].year,
                                  month = df.index[0].month,
                                  day = df.index[0].day,
                                  hour = 7, minute = 25, tz = TZ)

    end_dateTime = pd.Timestamp(year = df.index[-1].year,
                                month = df.index[-1].month,
                                day = df.index[-1].day,
                                hour = 17, minute = 35, tz = TZ)

    dateTime_index = pd.date_range(start_dateTime,
                                   end_dateTime,
                                   freq='5min').tolist()

    dateTime_index_df = pd.DataFrame()
    dateTime_index_df["Time"] = dateTime_index
    filtered_df = pd.merge_asof(dateTime_index_df, df,
                                on='Time',
                                direction='backward').set_index("Time").between_time('08:29','16:29')

    # slice 1: 8:30 to 10:30 predicting 12:30
    # slice 2: 10:30 to 12:30 predicting 14:30
    # slice 3: 12:30 to 14:30 predicting 16:30
    # slice 4: 14:30 to 16:30 predicting 08:30 +1


    # slice 1: starts between 9:30 and 10:30 at random
    # slice 2: starts between 10:30 and 11:30 at random
    # ...
    # slice n: starts before 14:00 to predict 16:00

    # separate data for premarket

    volumeset_list = []
    prev_v = None

    # change this to backfill with Nan instead of the previous value
    for v in filtered_df["Volume"]:

        if prev_v == None:
            if np.isnan(v):
                prev_v = 0
                volumeset_list.append(0)
            else:
                prev_v = v
                volumeset_list.append(v)

        elif prev_v != None:
            # if v == prev_v:
            #   volumeset_list.append(0)
            #   prev_v = v
            if np.isnan(v):
                volumeset_list.append(0)
                prev_v = 0
            else:
                volumeset_list.append(v)
                prev_v = v


    filtered_df["Volume"] = volumeset_list
    adjvolumeset_list = []

    prev_v = None

    # change this to backfill with Nan instead of the previous value
    for v in filtered_df["VolumeWeightedAvgPrice"]:
        if prev_v == None:
            if np.isnan(v):
                prev_v = 0
                adjvolumeset_list.append(0)
            else:
                prev_v = v
                adjvolumeset_list.append(v)
        elif prev_v != None:
            # if v == prev_v:
            #   adjvolumeset_list.append(0)
            #   prev_v = v
            if np.isnan(v):
                adjvolumeset_list.append(0)
                prev_v = 0
            else:
                adjvolumeset_list.append(v)
                prev_v = v

    filtered_df["VolumeWeightedAvgPrice"] = adjvolumeset_list

    preped_df = filtered_df.backfill()

    return preped_df


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return np.flip(np.rot90((arr.reshape(h//nrows, nrows, -1, ncols)
                             .swapaxes(1,2)
                             .reshape(-1, nrows, ncols)), axes = (1, 2)), axis = 1)
