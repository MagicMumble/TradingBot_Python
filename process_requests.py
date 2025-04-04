import http.server
import socketserver
import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from functions import create_data, round_indicator_value_to_2_signs_after_point
from model import reverse_one_hot

endpoint = '/data'
hostName = "localhost"
scaler_file = ''
model_file = ''


def process_request_body(json_string):
    df2 = pd.json_normalize(json_string)
    df2 = df2.set_index('Datetime')
    df2.index = pd.to_datetime(df2.index)
    df2 = df2.drop(labels=["ReqId"], axis=1)
    return df2


class MyServer(http.server.SimpleHTTPRequestHandler):
    old_data_prices = None

    def __init__(self, *args, directory=None, **kwargs):
        if directory is None:
            directory = os.getcwd()
        self.directory = os.fspath(directory)
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        super().__init__(*args, **kwargs)

    def normalize_one_datapoint(self, data, req_id):
        data_to_normalize = data.drop(labels=["Labels", "Adj Close"], axis=1)
        columns = data_to_normalize.columns

        data_scaled = self.scaler.transform(data_to_normalize.to_numpy())
        data_scaled = pd.DataFrame(data_scaled, columns=columns)
        data_scaled = round_indicator_value_to_2_signs_after_point(data_scaled)
        return data_scaled

    def send_200_response(self, results):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(results.encode(encoding='utf_8'))

    def process_request(self, req_id):
        try:
            logging.info('Start processing request')
            data = create_data(MyServer.old_data_prices, forex=False, create_one_datapoint=True).iloc[-1:, :]
            logging.info('Nil values found, changed to 0.0: %s', data.columns[data.isna().any()].tolist().__str__())

            # such indicators as cmo, cmfi cannot work with non changing values and often equal to 0 (na.nans)
            data.fillna(0.0, inplace=True)
            normalized_data = self.normalize_one_datapoint(data, req_id)
            logging.info(normalized_data)
            sample = np.array(normalized_data.values.tolist())

            row = sample.reshape(1, 15, 15, 1)
            prediction = self.model.predict(row)
            action = np.array(reverse_one_hot(prediction))[0]
            MyServer.old_data_prices = MyServer.old_data_prices.iloc[1:, :]
            return action
        except ValueError as e:
            info = f'{type(e)} - {str(e)}'
            logging.error('Error happened while processing the request: %s', info)
            return info

    def do_GET(self):
        if self.path.startswith(endpoint):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            logging.info(f"Get request: data = {data}")
            logging.info('Get request, id = %d', data['ReqId'])
            df = process_request_body(data)
            if MyServer.old_data_prices is None:
                MyServer.old_data_prices = df.copy()
                response = json.dumps({"RespId": data['ReqId'], "Action": 0})
            elif len(MyServer.old_data_prices) < 55:
                MyServer.old_data_prices = pd.concat([MyServer.old_data_prices, df])
                response = json.dumps({"RespId": data['ReqId'], "Action": 0})
            else:
                MyServer.old_data_prices = pd.concat([MyServer.old_data_prices, df])
                action = self.process_request(data['ReqId'])
                if type(action) == str:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = json.dumps({"RespId": data['ReqId'], "Error": action})
                    self.wfile.write(response.encode("utf-8"))
                    return

                # TODO: check the req id in requests and responses
                response = json.dumps({"RespId": data['ReqId'], "Action": int(action)})

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
            logging.info('Send response, id = %d', data['ReqId'])
            return

        self.send_error(404, 'Not found')


def start_server(port, model_file_new, scaler_file_new):
    global model_file, scaler_file
    model_file = model_file_new
    scaler_file = scaler_file_new

    webServer = socketserver.TCPServer((hostName, port), MyServer)
    logging.info(f"Server started http://{hostName}:{port}")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    logging.info("Server stopped.")
