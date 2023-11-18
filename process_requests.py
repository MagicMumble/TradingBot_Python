import http.server
import socketserver
import pandas as pd
import numpy as np
import os
import joblib
import json
from functions import create_data
from model import reverse_one_hot

PORT = 9999
endpoint = '/data'
hostName = "localhost"


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
        # TODO: add model and scaler file names to config
        self.model = joblib.load('prediction_tools/TCSG_minutes_model.sav')
        self.scaler = joblib.load('prediction_tools/TCSG_minutes_scaler.sav')
        super().__init__(*args, **kwargs)

    def normalize_one_datapoint(self, data, req_id):
        data_to_normalize = data.drop(labels=["Labels", "Adj Close"], axis=1)
        columns = data_to_normalize.columns

        data_scaled = self.scaler.transform(data_to_normalize.to_numpy())
        # index gets removed, columns are also not needed, might be removed too
        data_scaled = pd.DataFrame(data_scaled, columns=columns)
        return data_scaled

    def send_200_response(self, results):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(results.encode(encoding='utf_8'))

    def process_request(self, req_id):
        try:
            print('Start processing request')
            # TODO: check the scaler!
            data = create_data(MyServer.old_data_prices, forex=False, create_one_datapoint=True).iloc[-1:, :]
            print('Nil values found, changed to 0.0:', data.columns[data.isna().any()].tolist())

            # such indicators as cmo, cmfi cannot work with non changing values and often equal to 0 (na.nans)
            data.fillna(0.0, inplace=True)
            normalized_data = self.normalize_one_datapoint(data, req_id)
            print(normalized_data)
            sample = np.array(normalized_data.values.tolist())

            row = sample.reshape(1, 15, 15, 1)
            prediction = self.model.predict(row)
            action = np.array(reverse_one_hot(prediction))[0]
            MyServer.old_data_prices = MyServer.old_data_prices.iloc[1:, :]
            return action
        except ValueError as e:
            info = f'{type(e)} - {str(e)}'
            print('Error happened while processing the request: ', info)
            return info

    def do_GET(self):
        if self.path.startswith(endpoint):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            print(f"Get request: data = {data}")
            print('Get request, id =', data['ReqId'])
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
            print('Send response, id = ', data['ReqId'])
            return

        self.send_error(404, 'Not found')


def start_server():
    webServer = socketserver.TCPServer((hostName, PORT), MyServer)
    print(f"Server started http://{hostName}:{PORT}")

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
