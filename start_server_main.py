import numpy as np
import pandas as pd
import yfinance as yf
import math
from functions import *
from model import *
from trading_strategy import *
from historical_data import *
import matplotlib.pyplot as plt
import warnings
from process_requests import *

plt.style.use('seaborn-darkgrid')

# original code: https://github.com/omerbsezer/CNN-TA/tree/master

# TODO:5: ask Anton how they deal with high transaction fees? Do they use the constant ones?
# TODO:8: try working with daily data (every day price changes, not every minute)
# TODO:9: if i close the laptop, bot stops working. I should use the server that is always active.
# TODO:10: take into consideration the losses due to the small cable bandwidth, slow internet
# TODO:11: add stop-losses to the strategy (track if and when they are needed)
# TODO:12: think how to leverage the trades
# TODO:13: think about how to gather statistics about trades to optimize the strategy (in csv files), where to store it
# TODO:14: consider binance API and free BTC to USDT converts

if __name__ == '__main__':
    start_server()




