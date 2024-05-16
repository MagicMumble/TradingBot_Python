import unittest
import os

from historical_data import get_historical_data
from retrain_model_main import get_config
from datetime import datetime


class TestGettingHistoricalDataForTCSG(unittest.TestCase):
    def setUp(self):
        # is executed before each test method
        configPath = 'tests/config_test.yaml'
        configParams = get_config(configPath)
        self.filepath = configParams['dir_with_historical_data']
        self.token = configParams['token']
        self.target_api = configParams['target_api']
        self.ticker = 'TCSG'

    def test_get_historical_data_for_last_5_days(self):
        days = 5
        file_name = get_historical_data(self.ticker, self.filepath, days, self.token, self.target_api)
        self.assertTrue(os.path.isfile(file_name))
        self.assertFalse(os.path.getsize(file_name) == 0)

    def test_get_historical_data_within_period(self):
        from_ = datetime(2022, 11, 28)
        to = datetime(2022, 12, 5)
        file_name = get_historical_data(self.ticker, self.filepath, 0, self.token, self.target_api, from_=from_, to=to)
        self.assertTrue(os.path.isfile(file_name))
        self.assertFalse(os.path.getsize(file_name) == 0)

    def tearDown(self):
        # is executed after each test method
        pass


if __name__ == '__main__':
    unittest.main()

# to execute tests, run
# ~/PycharmProjects/TradingBot$ python3 -m unittest tests/historical_data_test/test_getting_historical_data.py
