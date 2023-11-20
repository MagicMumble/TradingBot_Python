import argparse
import yaml
from process_requests import *
import logging
from datetime import datetime


# original code: https://github.com/omerbsezer/CNN-TA/tree/master

# TODO:5: ask Anton how they deal with high transaction fees? Do they use the constant ones?
# TODO:8: try working with daily data (every day price changes, not every minute)
# TODO:9: if i close the laptop, bot stops working. I should use the server that is always active.
# TODO:10: take into consideration the losses due to the small cable bandwidth, slow internet
# TODO:11: add stop-losses to the strategy (track if and when they are needed)
# TODO:12: think how to leverage the trades
# TODO:13: think about how to gather statistics about trades to optimize the strategy (in csv files), where to store it
# TODO:14: consider binance API and free BTC to USDT converts

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Parses command line flags."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="A filepath to the config file"
    )
    return parser.parse_args(args)


def get_config(config_file):
    with open(config_file, 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def setup_logger():
    cur_time = datetime.now()
    logger_file_name = f'./logs/{cur_time}_startServer.log'
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S', level=logging.DEBUG,
                        filename=logger_file_name)


def main(args=None):
    logging.info('hello')
    args = parse_args(args)
    config_params = get_config(args.config)
    start_server(int(config_params['server_port']), config_params['model_file'], config_params['scaler_file'])


if __name__ == '__main__':
    setup_logger()
    # python3 start_server_main.py --config ./config.yaml 2>&1 | tee start_server.txt
    # python3 start_server_main.py --config ../config.yaml &> start_server0.txt
    main()
