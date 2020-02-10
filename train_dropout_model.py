import keras
import tensorflow as tf
import os
import argparse

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from project_utils import get_data, get_dimensions


def train_dropout_model(dataset, dropout_rate, train_epoch, load_pretrain=None):
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=True, percentage=1)
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
        model.add(Dropout(dropout_rate))
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
        model.add(Dropout(dropout_rate))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
    else:
        raise ValueError("%s is not a supported dataset!" % dataset)

    # load pretrained weights if apply
    if load_pretrain:
        model.load_weights(load_pretrain)

    # train
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=128, validation_data=(X_test, Y_test), nb_epoch=train_epoch, shuffle=True)

    # save weight
    if not os.path.exists('Model/%s_models/' % dataset):
        os.makedirs('Model/%s_models/' % dataset)
    weight_path = 'Model/%s_models/dropout_%.2f' % (dataset, dropout_rate)
    model.save_weights(weight_path)

    return weight_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR', type=str, help="dataset: CIFAR or MNIST")
    parser.add_argument('--dropout_rate', default=0.7, type=float, help="dropout rate in training")
    parser.add_argument('--train_epoch', default=70, type=int, help="train epochs")
    parser.add_argument('--load_pretrain', default='', type=str, help="path to load pretrain weight")

    args = parser.parse_args()
    train_dropout_model(dataset=args.dataset, dropout_rate=args.dropout_rate, train_epoch=args.train_epoch,
                        load_pretrain=args.load_pretrain)