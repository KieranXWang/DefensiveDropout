import tensorflow as tf
import keras
import numpy as np
import argparse

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from project_utils import get_data, get_dimensions


def defend_adversarial_attack(dataset, train_dropout_rate, test_dropout_rate, pretrain_dir, attack, epsilon,
                              test_samples, num_steps, step_size):
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # get data and dimensions
    img_rows, img_cols, img_channels = get_dimensions(dataset)
    [_, X_test, _, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=False, percentage=1)

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

    # make attack object
    if attack == 'FGSM':
        from attack_utils import FGSM
        attack = FGSM(model=model, epsilon=epsilon, dataset=dataset)
    elif attack == 'PGD':
        from attack_utils import PGD
        attack = PGD(model=model, num_steps=num_steps, step_size=step_size, epsilon=epsilon, dataset=dataset)
    elif attack == 'CWPGD':
        from attack_utils import CW_PGD
        attack = CW_PGD(model=model, num_steps=num_steps, step_size=step_size, epsilon=epsilon, dataset=dataset)
    else:
        raise ValueError('%s is not a valid attack name!' % attack)

    # perform attack
    result = []
    distortion = []

    for test_sample_idx in range(test_samples):
        print('generating adv sample for test sample ' + str(test_sample_idx))
        image = X_test[test_sample_idx:test_sample_idx + 1]
        label = Y_test[test_sample_idx:test_sample_idx + 1]

        for target in range(10):
            if target == label:
                continue

            target_input = np.array([target])
            adversarial = attack.perturb(image, target_input, sess)

            output = model.predict(adversarial)
            adv_pred = np.argmax(list(output)[0])
            result.append((adv_pred == target).astype(int))

            l_inf = np.amax(adversarial - image)
            distortion.append(l_inf)

    # compute attack success rate (ASR) and average distortion(L_inf)
    succ_rate = np.array(result).mean()
    mean_distortion = np.array(distortion).mean()

    print('Perform %s attack to dropout model[train_dropout=%.2f, test_dropout=%.2f]' % (attack, train_dropout_rate, test_dropout_rate))
    print('Attack succ rate (ASR) = %.4f' % succ_rate)
    print('Average distortion = %.2f' % mean_distortion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CIFAR', type=str, help="dataset: CIFAR or MNIST")
    parser.add_argument('--train_dropout_rate', default=0.7, type=float, help="dropout rate in training")
    parser.add_argument('--test_dropout_rate', default=0.7, type=float, help="dropout rate in testing")
    parser.add_argument('--pretrain_dir', default='Model/CIFAR_models/', type=str, help="dir to load pretrained weights.")
    parser.add_argument('--test_examples', default=10, help='number of test examples')
    parser.add_argument('--attack', default='CWPGD', help='FGSM, PGD or CWPGD')
    parser.add_argument('--epsilon', default=4, help='the L_inf bound of allowed adversarial perturbations, 8 means 8/255',
                        type=float)
    parser.add_argument('--num_steps', default=100, help='number of steps in generating adversarial examples, not work '
                                                         'for FGSM')
    parser.add_argument('--step_size', default=0.1, help='the step size in generating adversarial examples')

    args = parser.parse_args()
    defend_adversarial_attack(dataset=args.dataset, train_dropout_rate=args.train_dropout_rate,
                              test_dropout_rate=args.test_dropout_rate, pretrain_dir=args.pretrain_dir,
                              test_samples=args.test_examples, attack=args.attack, epsilon=args.epsilon / 255,
                              num_steps=args.num_steps, step_size=args.step_size)