import numpy as np
import pandas as pd

# create target from OHLC and Volume Data
def buildTargets(obs_array,
                 alph = 0.15,
                 volity_int = 10):

    """
    This function will take a complete set of train, val, and test
    data and return the targets. Volitility will be calculated over
    the 24 5min incriments. The Target shift is looking at 2 hours
    shift from current time

    shift_2hour = The amount of time the data interval take to equal 2 hours
                  (i.e. 5 min data interval is equal to 24)
    alph = The alpha value for calculating the shift in price
    volity_int = the number of incriments used to calculate volitility
    """

    target_close_list =[]

    for arr in obs_array:
        target_close_list.append(arr[3][-1])

    target_close_df = pd.DataFrame()
    target_close_df["Close"] = target_close_list

    returns = np.log(target_close_df['Close']/(target_close_df['Close'].shift()))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=(volity_int)).std()*np.sqrt(volity_int)

    # print(volatility[0:10])

    targets = [2] * len(target_close_df.Close)

    targets = np.where(target_close_df.Close.shift(-1) >= (target_close_df.Close * (1 + alph * volatility)),
                       1, targets)

    targets = np.where(target_close_df.Close.shift(-1) <= (target_close_df.Close * (1 - alph * volatility)),
                       0, targets)

    return targets


def get_class_distribution(obj):
    count_dict = {
        "up": 0,
        "flat": 0,
        "down": 0,
    }

    for i in obj:
        if i == 1:
            count_dict['up'] += 1
        elif i == 0:
            count_dict['down'] += 1
        elif i == 2:
            count_dict['flat'] += 1
        else:
            print("Check classes.")

    return count_dict


def buildTargets_VolOnly(full_df, volity_int=10):

    """
    This function will take a complete set of train, val, and test data and return the targets.
    Volitility will be calculated over the 252 5min incriments
    The Target shift is looking at 2 hours shift from current time
    """

    returns = np.log(full_df['Close']/(full_df['Close'].shift()))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=(volity_int)).std()*np.sqrt(volity_int)

    return volatility
