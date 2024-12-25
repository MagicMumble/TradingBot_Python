import logging

import joblib
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import shuffle


def reverse_one_hot(predictions):
    reversed_x = []
    for x in predictions:
        reversed_x.append(np.argmax(np.array(x)))
    return reversed_x


def train_cnn(training_df, test_df, params, filename, persist):
    """Trains and evaluates CNN on the given train and test data, respectively; saves model to the file"""

    logging.info("Training is starting ...")
    train_images = np.array(training_df.iloc[:, :-2].values.tolist())
    train_labels = training_df['Labels']
    train_prices = training_df['Adj Close']

    # should the data be shuffled? I think it should
    # if the data is reshuffled the tuning of hyperparameters for different models is unfair
    train_labels, train_prices = shuffle(train_labels, train_prices)

    test_images = np.array((test_df.iloc[:, :-2].values.tolist()))
    test_labels = test_df['Labels']
    test_prices = test_df['Adj Close']

    test_labels = tf.keras.utils.to_categorical(test_labels, params["num_classes"])
    train_labels = tf.keras.utils.to_categorical(train_labels, params["num_classes"])

    train_images = train_images.reshape(train_images.shape[0], params["input_w"], params["input_h"], 1)
    test_images = test_images.reshape(test_images.shape[0], params["input_w"], params["input_h"], 1)

    # CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(params["input_w"], params["input_h"], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params["num_classes"], activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy', 'mae', 'mse'])

    train_data_size = train_images.shape[0]
    test_data_size = test_images.shape[0]

    logging.info("model will be trained with {} and be tested with {} sample".format(train_data_size, test_data_size))
    logging.info("Fitting model to the training data...")
    # can add verbose=1 to the parameters
    model.fit(train_images, train_labels, batch_size=params["batch_size"], epochs=params["epochs"], verbose=1,
              validation_data=None)
    if persist:
        joblib.dump(model, filename)

    # can add verbose=1 to the parameters
    predictions_test = model.predict(test_images, batch_size=params["batch_size"], verbose=1)
    predictions_train = model.predict(train_images, batch_size=params["batch_size"], verbose=1)

    test_labels_for_metrics = np.array(reverse_one_hot(test_labels))
    train_labels_for_metrics = np.array(reverse_one_hot(train_labels))
    predictions_test_for_metrics = np.array(reverse_one_hot(predictions_test))
    predictions_train_for_metrics = np.array(reverse_one_hot(predictions_train))

    logging.info("Evaluation accuracy score (test)")
    test_acc_score = accuracy_score(test_labels_for_metrics, predictions_test_for_metrics)
    logging.info(test_acc_score)

    logging.info("Evaluation accuracy score (train)")
    logging.info(accuracy_score(train_labels_for_metrics, predictions_train_for_metrics))

    logging.info("Confusion matrix for the testing dataset")
    test_conf_matrix = confusion_matrix(test_labels_for_metrics, predictions_test_for_metrics)
    logging.info(test_conf_matrix)

    logging.info("Confusion matrix for the training dataset")
    logging.info(confusion_matrix(train_labels_for_metrics, predictions_train_for_metrics))

    logging.info("Classification report for the testing dataset")
    logging.info(classification_report(test_labels_for_metrics, predictions_test_for_metrics))

    logging.info("Classification report for the training dataset")
    logging.info(classification_report(train_labels_for_metrics, predictions_train_for_metrics))

    return predictions_test, test_labels, test_prices, test_acc_score, test_conf_matrix
