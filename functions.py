import yfinance as yf
import math
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import tulipy as ti
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
import time
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from sklearn.utils import shuffle
import warnings
import joblib
import logging
import os

warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO: experiment with the window size
def Labeling(data, windowSize):
    """
    Labels: 0 - HOLD, 1 - BUY, 2 - SELL
    """
    closePriceList = data['Adj Close'].values.tolist()
    counterRow, numberOfDaysInFile = 0, len(closePriceList)
    zeros_at_the_beginning = windowSize // 2
    labels = np.empty(zeros_at_the_beginning)
    labels.fill(np.nan)
    labels = labels.tolist()
    while counterRow < numberOfDaysInFile:
        min, max = math.inf, -math.inf
        counterRow += 1
        maxIndex, minIndex = -1, -1
        if counterRow >= windowSize:
            windowBeginIndex = counterRow - windowSize
            windowEndIndex = windowBeginIndex + windowSize - 1
            windowMiddleIndex = (windowBeginIndex + windowEndIndex) // 2
            for i in range(windowBeginIndex, windowEndIndex + 1):
                number = closePriceList[i]
                if number < min:
                    min = number
                    minIndex = i
                if number > max:
                    max = number
                    maxIndex = i
            if maxIndex == windowMiddleIndex:
                labels.append(2)
            elif minIndex == windowMiddleIndex:
                labels.append(1)
            else:
                labels.append(0)

    zeros = np.empty(windowSize - zeros_at_the_beginning - 1)
    zeros.fill(np.nan)
    labels = np.append(labels, zeros.tolist())
    new_data = data.copy()
    new_data['Labels'] = labels
    return new_data


################################ indicators ################################################


def ti_sma(period, data):
    res = ti.sma(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'SMA_{period}'] = add_nans(period - 1, res)
    return newFrame


def ti_ema(period, data):
    res = ti.ema(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'EMA_{period}'] = res
    return newFrame


def ti_hma(period, data):
    res = ti.hma(data['Close'].values, period=period)
    if period <= 8:
        change = 0
    elif period <= 15:
        change = 1
    elif period <= 24:
        change = 2
    else:
        change = 3
    newFrame = data.copy()
    newFrame[f'HMA_{period}'] = add_nans(period + change, res)
    return newFrame


def ti_wma(period, data):
    res = ti.wma(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'WMA_{period}'] = add_nans(period - 1, res)
    return newFrame


def ti_triple_ema(period, data):
    res = ti.tema(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'T_EMA_{period}'] = add_nans(period * 3 - 3, res)
    return newFrame


def add_nans(period, res):
    a = np.empty(period)
    a.fill(np.nan)
    return np.append(a, res)


def ti_willr(period, data):
    res = ti.willr(data['High'].values, data['Low'].values, data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'WILLR_{period}'] = add_nans(period - 1, res)
    return newFrame


def ti_rsi(period, data):
    res = ti.rsi(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'RSI_{period}'] = add_nans(period, res)
    return newFrame


def ti_cci(period, data):
    res = ti.cci(data['High'].values, data['Low'].values, data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'CCI_{period}'] = add_nans(period + period - 2, res)
    return newFrame


def ti_cmo(period, data):
    res = ti.cmo(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'CMO_{period}'] = add_nans(period, res)
    return newFrame


def ti_roc(period, data):
    res = ti.roc(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'ROC_{period}'] = add_nans(period, res)
    return newFrame


def ti_kama(period, data):
    """
    ti_kama is an extra indicator for forex data to make the image of size 15x15 (because we can't use cmf indicator
    for the forex data)
    """
    res = ti.kama(data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'KAMA_{period}'] = add_nans(period - 1, res)
    return newFrame


def ti_dmi(period, data):
    res = ti.dx(data['High'].values, data['Low'].values, data['Close'].values, period=period)
    newFrame = data.copy()
    newFrame[f'DMI_{period}'] = add_nans(period - 1, res)
    return newFrame


def get_periods(period):
    return period, period * 2, int(0.75 * period)


def ti_macd(period, data):
    shortTermEma = ti.ema(data['Close'].values, period=period)
    longTermEma = ti.ema(data['Close'].values, period=period * 2)
    res = shortTermEma - longTermEma
    newFrame = data.copy()
    newFrame[f'MACD_{period}'] = res
    return newFrame


def ti_ppo(period, data):
    short_period, long_period, _ = get_periods(period)
    res = ti.ppo(data['Close'].values, short_period, long_period)
    newFrame = data.copy()
    newFrame[f'PPO_{period}'] = add_nans(1, res)
    return newFrame


def ti_cmfi(period, data):
    """
    etfs have volumes, forex data doesn't, for the forex data we use kama indicator instead
    """
    res = ti.mfi(data['High'].values, data['Low'].values, data['Close'].values, data['Volume'].values, period=period)
    newFrame = data.copy()
    newFrame[f'CMFI_{period}'] = add_nans(period, res)
    return newFrame


def ti_psar(period, data):
    res, data_high, data_low = [np.NAN], data['High'].values, data['Low'].values
    for i in range(1, len(data)):
        start = max(0, i - period + 1)
        res.append(ti.psar(data_high[start:i + 1], data_low[start:i + 1], 0.02, 0.2)[-1])
    newFrame = data.copy()
    newFrame[f'PSAR_{period}'] = res
    return newFrame


def adjust_the_prices(data):
    """
    there is no volume records for forex data, so we skip this step; but shouldn't skip for etf data
    """
    data['Open'] = (data['Open'] * data['Volume']) / data['Adj Close']
    data['High'] = (data['High'] * data['Volume']) / data['Adj Close']
    data['Low'] = (data['Low'] * data['Volume']) / data['Adj Close']
    # data.rename(columns={"Adj Close": "Close"}, inplace=True)
    return data


def create_data(data, forex=True, create_one_datapoint=False):
    if not forex:
        data = adjust_the_prices(data)

    data = data.astype('float64')

    if not create_one_datapoint:
        data = Labeling(data, 11)
    else:
        # add mock labels
        labels = np.empty(len(data))
        labels.fill(0)
        data['Labels'] = labels

    # order of indicators matters
    etf_indicators = [ti_rsi, ti_willr, ti_wma, ti_ema, ti_sma, ti_hma, ti_triple_ema, ti_cci, ti_cmo, ti_macd, ti_ppo,
                      ti_roc, ti_cmfi, ti_dmi, ti_psar]

    forex_indicators = [ti_rsi, ti_willr, ti_wma, ti_ema, ti_sma, ti_kama, ti_hma, ti_triple_ema, ti_cci, ti_cmo,
                        ti_macd, ti_ppo, ti_roc, ti_dmi, ti_psar]

    # not forex
    absolute_indicators = [ti_rsi, ti_willr, ti_cci, ti_cmo, ti_ppo, ti_roc, ti_dmi]

    # ranges of absolute indicators
    # rsi - [0, 100]
    # willr - [-100, 0]
    # cci = [-inf, inf], but absolute
    # cmo = [-inf, inf], but absolute
    # ppo, roc, dmi

    for period in range(5, 20):
        if forex:
            indicators = forex_indicators
        else:
            # indicators = eft_indicators
            # for absolute indicators the time period of testing data should be longer and there is no need to retrain
            # the model (only for the new patterns in the market)
            indicators = absolute_indicators
        for indicator in indicators:
            data = indicator(period, data)

    # do not remove 'Adj Close'
    data.drop(labels=["Volume", "Low", "High", "Open", "Close"], axis=1, inplace=True)

    if not create_one_datapoint:
        data.dropna(inplace=True)

    return data


def read_data_with_multiple_files(directory, ticket, forex):
    datas = []
    for file in sorted(os.listdir(directory)):
        if ticket in file:
            data = pd.read_csv(os.path.join(directory, file), index_col=['Datetime'])
            data.index = pd.to_datetime(data.index)
            data = create_data(data, forex)
            datas.append(data)
    data = datas[0]
    for i in range(1, len(datas)):
        data = data.append(datas[i])
    return data


def read_data_from_single_file(file_name, forex):
    data = pd.read_csv(file_name, index_col=['Datetime'])
    data.index = pd.to_datetime(data.index)
    return create_data(data, forex)


def save_to_csv():
    start = ['2023-08-20', '2023-08-27', '2023-09-03', '2023-09-10']
    end = ['2023-08-27', '2023-09-03', '2023-09-10', '2023-09-17']
    currencies = ['EURUSD=X', 'EURTRY=X', 'EURRUB=X', 'ARSUSD=X', 'AUDUSD=X', 'JPYUSD=X', 'IRRUSD=X', 'KPW=X',
                  'EURDKK=X', 'SEKUSD=X']
    etfs = ['XLF', 'QQQ', 'SPY', 'XLP', 'EWZ', 'EWH', 'XLY', 'XLE']
    files = os.listdir('./historical_data/forex_minutes') + (os.listdir('./historical_data/etfs_minutes'))

    for i in range(4):
        for j in range(len(currencies)):
            file_name = f'{start[i]}_to_{end[i]}_{currencies[j]}.csv'
            if file_name not in files:
                data = yf.download(currencies[j], start=start[i], end=end[i], interval='1m')
                if len(data) > 0:
                    data.to_csv('./historical_data/forex_minutes/' + file_name)
            else:
                logging.info('already exists: %s', file_name)
        for j in range(len(etfs)):
            file_name = f'{start[i]}_to_{end[i]}_{etfs[j]}.csv'
            if file_name not in files:
                data = yf.download(etfs[j], start=start[i], end=end[i], interval='1m')
                if len(data) > 0:
                    data.to_csv('./historical_data/etfs_minutes/' + file_name)
            else:
                logging.info('already exists: %s', file_name)


def round_indicator_value_to_2_signs_after_point(data):
    return data.round(2)


def normalize(data, filename):
    scaler = MinMaxScaler((-1, 1))

    # Adj Close is never added back, we don't need it
    data_to_normalize = data.drop(labels=["Labels", "Adj Close"], axis=1)
    columns = data_to_normalize.columns

    # -212.65458552306788 109066.66666666667 (17221, 225) - min, max, shape
    # data_to_normalize.to_csv('prediction_tools/data_to_normalize_whole.csv')

    data_scaled = scaler.fit_transform(data_to_normalize.to_numpy())
    joblib.dump(scaler, filename)

    data_scaled = pd.DataFrame(data_scaled, columns=columns)
    data_scaled = round_indicator_value_to_2_signs_after_point(data_scaled)
    data_scaled['Labels'] = data['Labels'].values  # use further as a label for training/testing data
    data_scaled['Adj Close'] = data['Adj Close'].values # needed to test the profit based on predictions

    return data_scaled


def calculate_class_ratios(data):
    l0_train = data.loc[data['Labels'] == 0]
    l1_train = data.loc[data['Labels'] == 1]
    l2_train = data.loc[data['Labels'] == 2]
    l0_size = l0_train.shape[0]
    l1_size = l1_train.shape[0]
    l2_size = l2_train.shape[0]

    l0_l1_ratio = (l0_size // l1_size)
    l0_l2_ratio = (l0_size // l2_size)
    logging.info("l0_size: %d, l1_size: %d, l2_size: %d", l0_size, l1_size, l2_size)
    logging.info("l0_l1_ratio: %f, l0_l2_ratio: %f", l0_l1_ratio, l0_l2_ratio)
    return l0_l1_ratio, l0_l2_ratio


def oversample_manual_data(data):
    st = time.time()
    data = data.iloc[15:, :]
    logging.info("Before")
    l0_l1_ratio, l0_l2_ratio = calculate_class_ratios(data)

    l1_new = pd.DataFrame()
    l2_new = pd.DataFrame()
    for idx, row in data.iterrows():
        if row['Labels'] == 1:
            for i in range(l0_l1_ratio):
                l1_new = l1_new.append(row)
        if row['Labels'] == 2:
            for i in range(l0_l2_ratio):
                l2_new = l2_new.append(row)

    data = data.append(l1_new)
    data = data.append(l2_new)

    data = shuffle(data)
    logging.info("After")
    calculate_class_ratios(data)
    elapsed_time = time.time() - st
    logging.info('Oversampling (manual) time: %f minutes', elapsed_time / 60)

    # rounding prices to two values after point
    data_to_oversample = data.drop(labels=["Labels"], axis=1)
    columns = data_to_oversample.columns
    data_to_oversample = round_indicator_value_to_2_signs_after_point(data_to_oversample)

    data_scaled = pd.DataFrame(data_to_oversample, columns=columns)
    data_scaled['Labels'] = data['Labels'].values

    return data_scaled


def oversample_ADASYN_data(data):
    # oversampling technique - adaptive synthetic sampling
    # generate more synthetic examples in regions of the feature space where the density of minority examples is low,
    # and fewer or none where the density is high.

    data_to_oversample = data.drop(labels=["Labels"], axis=1)
    columns = data_to_oversample.columns
    labels = data["Labels"].to_numpy()
    data_to_oversample = round_indicator_value_to_2_signs_after_point(data_to_oversample)

    sm = ADASYN(random_state=40, n_jobs=8, sampling_strategy='not majority')
    st = time.time()
    X_train_resampled_ADASYN, y_train_resampled_ADASYN = sm.fit_resample(data_to_oversample, labels)
    elapsed_time = time.time() - st

    data_resampled = pd.DataFrame(X_train_resampled_ADASYN, columns=columns)
    data_resampled['Labels'] = y_train_resampled_ADASYN
    data_resampled = data_resampled.copy()

    logging.info('Oversampling (ADASYN) time: %f minutes', elapsed_time / 60)
    logging.info('Original dataset shape (train only): %s', Counter(labels).__str__())
    logging.info('Resampled dataset shape (train only): %s', Counter(y_train_resampled_ADASYN).__str__())

    return data_resampled


def undersample_data(data, n_init=3):
    data_to_oversample = data.drop(labels=["Labels"], axis=1)
    columns = data_to_oversample.columns
    labels = data["Labels"].to_numpy()
    data_to_oversample = round_indicator_value_to_2_signs_after_point(data_to_oversample)

    # TODO: check how many clusters I should specify with the n_init parameter
    cc = ClusterCentroids(estimator=MiniBatchKMeans(n_init=n_init), sampling_strategy='not minority')
    st = time.time()
    X_resampled_train, y_resampled_train = cc.fit_resample(data_to_oversample, labels)
    elapsed_time = time.time() - st

    data_resampled = pd.DataFrame(X_resampled_train, columns=columns)
    data_resampled['Labels'] = y_resampled_train
    data_resampled = data_resampled.copy()

    logging.info('Undersampling (ClusterCentroids) time: %f minutes', elapsed_time / 60)
    logging.info('Original dataset shape (train only): %s', Counter(labels).__str__())
    logging.info('Resampled dataset shape (train only): %s', Counter(y_resampled_train).__str__())

    return data_resampled


def calculate_volatility(data):
    data['Daily Return'] = ""
    for i in range(1, len(data)):
        data['Daily Return'][i] = np.log(data['Adj Close'][i] / data['Adj Close'][i - 1])

    # because the prices are measured every week (for day trading);
    # for different time frequency it could be daily volatility
    minuteVolatility = statistics.stdev(data['Daily Return'][1:])
    logging.info("The minute volatility is: {:.2%}".format(minuteVolatility))

    exchange_working_hours_per_week = 40
    weeklyVolatility = minuteVolatility * np.sqrt(exchange_working_hours_per_week * 60)
    logging.info("The weekly volatility is: {:.2%}".format(weeklyVolatility))


def plot_prices(data):
    plt.figure(figsize=(15, 7))
    data['Adj Close'].plot()

    plt.title('EUR/USD Data', fontsize=16)
    plt.xlabel('Year-Month', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(['Close'], prop={'size': 15})
    plt.show()
