from tinkoff.invest.constants import INVEST_GRPC_API, INVEST_GRPC_API_SANDBOX
from datetime import timedelta
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from retrain_model_main import parse_args, get_config
import logging
import os
import csv

DAYS_OF_DATA = 30


def get_price(price):
    return price.units + float(f"0.{str(price.nano)[:2]}")


def get_historical_data(ticker, filepath, days, token, target_api, from_=now(), to=now()):
    """
    A wrapper for getting historical data - within the period or the data for the last days ending today
    """

    # TODO: pay attention to when the request to get historical data is sent!
    # exchange works from 7.00 to 20.00, no weekends (timezone +00)
    # from 9.00 to 22.00 in my timezone (EU, +02)
    # from 10.00 to 23.00 in moscow timezone (+03)

    if days == 0:
        logging.info('Get historical data from %t to %t', from_, to)
        return get_historical_data_within_period(ticker, filepath, token, target_api, from_, to)
    else:
        logging.info('Get historical data for the last %d days period', days)
        from_ = now() - timedelta(days=days)
        to = now()
        return get_historical_data_within_period(ticker, filepath, token, target_api, from_, to)


def get_figi(instruments):
    for instrument in instruments:
        # instrument.class_code == 'TQBR'-торгуемый класс, SPEQ - неторгуемый
        # now only working with the stocks, but there are other types:
        #     INSTRUMENT_TYPE_UNSPECIFIED = 0
        #     INSTRUMENT_TYPE_BOND = 1
        #     INSTRUMENT_TYPE_SHARE = 2
        #     INSTRUMENT_TYPE_CURRENCY = 3
        #     INSTRUMENT_TYPE_ETF = 4
        #     INSTRUMENT_TYPE_FUTURES = 5
        #     INSTRUMENT_TYPE_SP = 6
        #     INSTRUMENT_TYPE_OPTION = 7
        #     INSTRUMENT_TYPE_CLEARING_CERTIFICATE = 8
        if instrument.api_trade_available_flag and instrument.instrument_type == 'share' and instrument.class_code == 'TQBR':
            return instrument.figi


def get_historical_data_within_period(ticker, filepath, token, target_api, from_, to):
    """
    Saves prices and volume data for the last month with the one-minute frequency
    :param ticker: ticker of derivative
    :param filepath: directory to save the csv file with data
    """

    # let's make sure that the directory is empty before saving new historical data in it
    for file in sorted(os.listdir(filepath)):
        logging.info('deleted file: %s', filepath + file)
        os.remove(filepath + file)

    with Client(token, target=target_api) as client:
        r = client.instruments.find_instrument(query=ticker)
        figi = get_figi(r.instruments)

        logging.info('FIGI: %s', figi)
        candles = client.get_all_candles(
            figi=figi,
            from_=from_,
            to=to,
            interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
        )
        processed_candles = [[candle.time, get_price(candle.open), get_price(candle.high), get_price(candle.low),
                              get_price(candle.close), get_price(candle.close), candle.volume]
                             for candle in candles]
        logging.info(f'Loaded historical data for {ticker}, data size is {len(processed_candles)}')

        header = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

        file_name = f'{filepath}/{from_.date()}_to_{to.date()}_{ticker}.csv'
        with open(file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for line in processed_candles:
                writer.writerow(line)
    return file_name


def main(args=None):
    args = parse_args(args)
    configPath = args.config
    configParams = get_config(configPath)
    get_historical_data('TCSG', configParams['dir_with_historical_data'], DAYS_OF_DATA, configParams['token'],
                        configParams['target_api'])


if __name__ == "__main__":
    main()
