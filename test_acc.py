import keras
import numpy as np
import argparse

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from project_utils import get_data, get_dimensions


def test_acc(dataset, train_dropout_rate, test_dropout_rate, pretrain_dir):
    [_, X_test, _, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=False, percentage=1)
    img_rows, img_cols, img_channels = get_dimensions(dataset)

    # model
    keras.backend.set_learning_phase(1)
    if dataset == 'CIFAR':
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, img_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(test_dropout_rate))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(10))
    elif dataset == 'MNIST':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(img_rows, img_cols, img_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(test_dropout_rate))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
    else:
        raise ValueError("%s is not a supported dataset!" % dataset)

    # load pretrained weights of the train_dropout_rate
    weight_file = 'dropout_%.2f' % train_dropout_rate
    weight_path = pretrain_dir + weight_file
    model.load_weights(weight_path)

    # test acc
    # note: it is more accurate to feed data points one by one, because of the randomness of the model
    # PS: you don't want to get the acc just for a single model realization
    score = []
    for i in range(X_test.shape[0]):
        x = X_test[i:i + 1]
        y = Y_test[i]
        # pred = keras_model.predict(x)
        pred = np.argmax(model.predict(x)[0])
        if np.array_equal(y, pred):
            score.append(1)
        else:
            score.append(0)

    acc = np.mean(np.array(score))

    print('Test Acc. of Model[test_dropout=%.2f, train_dropout=%.2f] is %.4f' % (test_dropout_rate, train_dropout_rate, acc))
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR', type=str, help="dataset: CIFAR or MNIST")
    parser.add_argument('--train_dropout_rate', default=0.7, type=float, help="dropout rate in training")
    parser.add_argument('--test_dropout_rate', default=0.7, type=float, help="dropout rate in testing")
    parser.add_argument('--pretrain_dir', default='Model/CIFAR_models/', type=str, help="dir to load pretrained weights.")

    args = parser.parse_args()
    test_acc(dataset=args.dataset, train_dropout_rate=args.train_dropout_rate, test_dropout_rate=args.test_dropout_rate,
             pretrain_dir=args.pretrain_dir)


