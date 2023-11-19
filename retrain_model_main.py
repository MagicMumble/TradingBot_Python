import argparse
import time

from functions import *
from model import *
from trading_strategy import *
import matplotlib.pyplot as plt
from process_requests import *
from historical_data import *
from datetime import date
from datetime import datetime
import yaml

plt.style.use('seaborn-darkgrid')

# original code: https://github.com/omerbsezer/CNN-TA/tree/master

# TODO:5: ask Anton how they deal with high transaction fees? Do they use the constant ones?
# TODO:7: investigate the etfs' volatility which are predicted with the best accuracy
# TODO:8: try working with daily data (every day price changes, not every minute)
# TODO:9: if i close the laptop, bot stops working. I should use the server that is always active.
# TODO:10: take into consideration the losses due to the small cable bandwidth, slow internet
# TODO:13: think about how to gather statistics about trades to optimize the strategy (in csv files), where to store it
# TODO:17: what to do if the file with statistics is too large? It shouldn't keep writing to the same file every time;
# I should create a script that will restart a retraining process specifying different output file maybe once a month
# TODO:18: create a deployment script + venv or conda env to deploy the code on a remote server

TICKET_NAME = 'TCSG'
TOKEN = ''
DIR_WITH_HISTORICAL_DATA = ''
SCALER_FILE = ''
MODEL_FILE = ''
TRAINING_PROPORTION = 0.7
DAYS_OF_DATA = 30

# manual, ADASYN - oversampling methods, ClusterCentroids - under-sampling method (requires more data)
# cannot use more data for under-sampling cause the values of last year are not relevant for the current
# month, and we need to get the data for over 15 months period to get an equal amount that we end up working
# on when using over sampling techniques; but we could use different indicators whose values do not depend
# on the actual stock values (for example the indicators with the values within 0-100 range)
POSSIBLE_ALGORITHMS_TO_BALANCE_THE_CLASSES = ['ADASYN', 'manual', 'ClusterCentroids']
ALGORITHM_TO_BALANCE_THE_CLASSES = 'ClusterCentroids'
N_INIT = 3

params = {"input_w": 15, "input_h": 15, "num_classes": 3, "batch_size": 32, "epochs": 2000}


def train_best_model(training_set, testing_set):
    print('Model retraining started')

    if ALGORITHM_TO_BALANCE_THE_CLASSES == 'ADASYN':
        training_set = oversample_ADASYN_data(training_set)
    elif ALGORITHM_TO_BALANCE_THE_CLASSES == 'manual':
        training_set = oversample_manual_data(training_set)
    elif ALGORITHM_TO_BALANCE_THE_CLASSES == 'ClusterCentroids':
        training_set = undersample_data(training_set, n_init=N_INIT)
    else:
        print(ALGORITHM_TO_BALANCE_THE_CLASSES + 'algorithm is not known as an algorithm to solve the imbalanced '
                                                 'classes problem. Back up to the ClusterCentroids algorithm')
        training_set = undersample_data(training_set)

    print("train_df size: ", training_set.shape)

    # TODO: should round values to two values after point??
    train_and_test_model(training_set, testing_set)


def split_into_train_and_test():
    file_name = get_historical_data(TICKET_NAME, DIR_WITH_HISTORICAL_DATA, DAYS_OF_DATA, TOKEN)  # - tinkoff invest

    # data = read_data(DIR_WITH_HISTORICAL_DATA, 'EURUSD=X', forex=True)
    # data = read_data(DIR_WITH_HISTORICAL_DATA, 'EWH', forex=False)
    data = read_data_from_single_file(file_name, forex=False)

    normalized_data = normalize(data, SCALER_FILE)

    # we used 4 files for prediction so testing set contains one week long data set
    training_proportion = int(TRAINING_PROPORTION * len(normalized_data))
    training_set = normalized_data[:training_proportion]
    testing_set = normalized_data[training_proportion:]
    return training_set, testing_set


def train_and_test_model(training_set, testing_set):
    predictions, test_labels, test_prices, test_acc_score, test_conf_matrix = train_cnn(training_set, testing_set,
                                                                                        params, MODEL_FILE)
    labels = np.argmax(predictions, axis=1).tolist()
    # strategy(labels, test_prices.tolist(), 1, constant_transaction_fee=True)
    profit = strategy(labels, test_prices.tolist(), 0, constant_transaction_fee=True)
    # strategy(labels, test_prices.tolist(), 0.005, constant_transaction_fee=False)
    print('Model retraining finished')
    return profit, test_acc_score, test_conf_matrix


def tune_hyperparameters_for_undersampling(training_set, testing_set):
    start = time.time()
    print("Start hyperparameters tuning for undersampling")
    n_init = [1, 2, 3]
    batch_size = [64, 256]
    epochs = [300, 600, 900, 1200]

    training_stage = []

    for n in n_init:
        training_set_undersampled = undersample_data(training_set, n_init=n)
        print("train_df size: ", training_set_undersampled.shape)
        for bs in batch_size:
            params['batch_size'] = bs
            for e in epochs:
                params['epochs'] = e
                print('PARAMETERS:', n, bs, e)
                profit, test_acc_score, test_conf_matrix = train_and_test_model(training_set_undersampled, testing_set)
                training_stage.append((test_acc_score, test_conf_matrix, n, bs, e, profit))

    n_new, batch_size_new, epochs_new, profit = get_best_model_params(training_stage)
    print("BEST PARAMETERS:", n_new, batch_size_new, epochs_new, profit)
    print('Model retuning and retraining finished for undersampling:', time.time() - start, 'seconds')
    return 'ClusterCentroids', (profit, n_new, batch_size_new, epochs_new)


def get_best_model_params(parameters):
    # sort by the highest profit
    parameters = sorted(parameters, key=lambda x: -x[5])[:5]

    # sort by the minimal number of negative 1 that predicted as 2 and 2 that predicted as 1 (the sum of them)
    parameters = sorted(parameters, key=lambda x: x[1][1][2] + x[1][2][1])
    i = 1
    min_false_negatives_num = parameters[0][1][1][2] + parameters[0][1][2][1]
    while parameters[i][1][1][2] + parameters[i][1][2][1] < min_false_negatives_num + 10 and i < len(parameters) - 1:
        i += 1

    # sort by the highest accuracy score for the testing dataset
    parameters = sorted(parameters[:i], key=lambda x: -x[0])
    return parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]


def tune_hyperparameters_for_ADASYN(training_set, testing_set):
    start = time.time()
    print("Start hyperparameters tuning for ADASYN")
    result = tune_hyperparameters_for_oversampling(oversample_ADASYN_data, training_set, testing_set)
    print('Model retuning and retraining finished for ADASYN:', time.time() - start, 'seconds')
    return 'ADASYN', result


def tune_hyperparameters_for_manual(training_set, testing_set):
    start = time.time()
    print("Start hyperparameters tuning for manual oversampling")
    result = tune_hyperparameters_for_oversampling(oversample_manual_data, training_set, testing_set)
    print('Model retuning and retraining finished for manual oversampling:', time.time() - start, 'seconds')
    return 'manual', result


def tune_hyperparameters_for_oversampling(oversample_function, training_set, testing_set):
    batch_size = [64, 256]
    epochs = [600, 900, 1200]

    training_set = oversample_function(training_set)
    training_stage = []
    print("train_df size: ", training_set.shape)

    for bs in batch_size:
        params['batch_size'] = bs
        for e in epochs:
            params['epochs'] = e
            print('PARAMETERS:', bs, e)
            profit, test_acc_score, test_conf_matrix = train_and_test_model(training_set, testing_set)
            training_stage.append((test_acc_score, test_conf_matrix, -1, bs, e, profit))

    n_new, batch_size_new, epochs_new, profit = get_best_model_params(training_stage)
    print("BEST PARAMETERS:", n_new, batch_size_new, epochs_new, profit)
    return profit, n_new, batch_size_new, epochs_new


def retrain_on_weekends():
    checked = False
    while True:
        # 5 - Saturday, 6 - Sunday, 0 - Monday
        # the model gets retrained every Saturday
        if not checked and (date.today().weekday() == 5 or date.today().weekday() == 6):
            print("Start general tuning", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            start_time = time.time()
            training_set, testing_set = split_into_train_and_test()

            best_pars = (tune_hyperparameters_for_undersampling(training_set, testing_set),
                         tune_hyperparameters_for_ADASYN(training_set, testing_set),
                         tune_hyperparameters_for_manual(training_set, testing_set))
            # sort by profit
            best_pars = sorted(best_pars, key=lambda x: -x[1][0])[0]
            algo_name, n_new, batch_size_new, epochs_new = best_pars[0], best_pars[1][1], best_pars[1][2], \
                best_pars[1][3]

            # train the best model
            global ALGORITHM_TO_BALANCE_THE_CLASSES, N_INIT
            ALGORITHM_TO_BALANCE_THE_CLASSES = algo_name
            N_INIT = n_new
            params['batch_size'], params['epochs'] = batch_size_new, epochs_new
            print("CHOSEN PARAMETERS:", algo_name, n_new, batch_size_new, epochs_new)
            train_best_model(training_set, testing_set)
            print("TOTAL TIME (general tuning):", time.time() - start_time, 'seconds')
            checked = True
        else:
            checked = False
            # wait for 1 hour
            time.sleep(1 * 60 * 60)


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


def changeGlobalVars(config_params):
    global TOKEN, DIR_WITH_HISTORICAL_DATA, SCALER_FILE, MODEL_FILE
    TOKEN = config_params['token_sandbox']
    DIR_WITH_HISTORICAL_DATA = config_params['dir_with_historical_data']
    SCALER_FILE = config_params['scaler_file']
    MODEL_FILE = config_params['model_file']


def main(args=None):
    print('start in production')
    args = parse_args(args)
    config_params = get_config(args.config)
    changeGlobalVars(config_params)
    retrain_on_weekends()


def test(args=None):
    print('start in test')
    args = parse_args(args)
    config_params = get_config(args.config)
    changeGlobalVars(config_params)
    training_set, testing_set = split_into_train_and_test()
    train_best_model(training_set, testing_set)


def test_hyperparameters_tuning(args=None):
    print('start in hyperparameters tuning test')
    args = parse_args(args)
    config_params = get_config(args.config)
    changeGlobalVars(config_params)
    training_set, testing_set = split_into_train_and_test()
    tune_hyperparameters_for_undersampling(training_set, testing_set)

    # tune_hyperparameters_for_ADASYN(training_set, testing_set)
    # tune_hyperparameters_for_manual(training_set, testing_set)


if __name__ == '__main__':
    # start in production
    # python3 retrain_model_main.py --config ./config.yaml 2>&1 | tee production_test2.txt
    main()

    # test the data reading and model training
    # python3 retrain_model_main.py --config ./config.yaml 2>&1 | tee test.txt
    # test()

    # tune the hyperparameters
    # python3 retrain_model_main.py --config ./config.yaml 2>&1 | tee tuningParameters_undersampling.txt
    # test_hyperparameters_tuning()

# -------------------------------------------------------
# check statistics:
# cat tuningParameters_undersampling.txt | grep -E "PARAMETERS:|accuracy score|Confusion matrix|Classification report" -A 10 | grep "PARAMETERS: 1 256 300" <best params here> -A 50
# without -1 as a value for n in the results for oversampling techniques (ADASYN and manual)
# tuning results:

# ADASYN:
# BEST PARAMETERS: -1 256 900
# Evaluation accuracy score (test)
# 0.7141529885413441
# Evaluation accuracy score (train)
# 0.34594291779691
# Confusion matrix for the testing dataset
# [[2272   35  514]
#  [ 167    5   29]
#  [ 170    8   29]]
# Confusion matrix for the training dataset
# [[4581  952 5783]
#  [4475 1066 5703]
#  [4362  908 6086]]
# Classification report for the testing dataset
#               precision    recall  f1-score   support
#
#            0       0.87      0.81      0.84      2821
#            1       0.10      0.02      0.04       201
#            2       0.05      0.14      0.07       207
#
#     accuracy                           0.71      3229
#    macro avg       0.34      0.32      0.32      3229
# weighted avg       0.77      0.71      0.74      3229

# Our System => totalMoney = 10317.18

# ----------------------------------------------------------------------

# undersampling (ClusterCentroids)
# BEST PARAMETERS: 1 256 300
# Evaluation accuracy score (test)
# 0.8250232270052648
# Evaluation accuracy score (train)
# 0.3572984749455338
# Confusion matrix for the testing dataset
# [[2645   34  142]
#  [ 195    5    1]
#  [ 192    1   14]]
# Confusion matrix for the training dataset
# [[431 188 146]
#  [377 229 159]
#  [389 216 160]]
# Classification report for the testing dataset
#               precision    recall  f1-score   support
#
#            0       0.87      0.94      0.90      2821
#            1       0.12      0.02      0.04       201
#            2       0.09      0.07      0.08       207
#
#     accuracy                           0.83      3229
#    macro avg       0.36      0.34      0.34      3229
# weighted avg       0.78      0.83      0.80      3229
#
# Classification report for the training dataset
#               precision    recall  f1-score   support
#
#            0       0.36      0.56      0.44       765
#            1       0.36      0.30      0.33       765
#            2       0.34      0.21      0.26       765
#
#     accuracy                           0.36      2295
#    macro avg       0.36      0.36      0.34      2295
# weighted avg       0.36      0.36      0.34      2295

# Our System => totalMoney = 10025.99

# ----------------------------------------------------------------------

# manual (oversampling):
# BEST PARAMETERS: -1 256 900
# Evaluation accuracy score (test)
# 0.36079281511303807
# Evaluation accuracy score (train)
# 0.3464211965215368
# Confusion matrix for the testing dataset
# [[ 997 1397  427]
#  [  86   93   22]
#  [  61   71   75]]
# Confusion matrix for the training dataset
# [[1155 7603 2545]
#  [1030 7947 2643]
#  [1009 7642 2809]]
# Classification report for the testing dataset
#               precision    recall  f1-score   support
#
#            0       0.87      0.35      0.50      2821
#            1       0.06      0.46      0.11       201
#            2       0.14      0.36      0.21       207
#
#     accuracy                           0.36      3229
#    macro avg       0.36      0.39      0.27      3229
# weighted avg       0.77      0.36      0.46      3229
#
# Classification report for the training dataset
#               precision    recall  f1-score   support
#
#            0       0.36      0.10      0.16     11303
#            1       0.34      0.68      0.46     11620
#            2       0.35      0.25      0.29     11460
#
#     accuracy                           0.35     34383
#    macro avg       0.35      0.34      0.30     34383
# weighted avg       0.35      0.35      0.30     34383

# Our System => totalMoney = 10095.39
