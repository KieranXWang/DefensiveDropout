# Defensive Dropout

Defensive Dropout is a simple and easy-configurable stochastic defense method against adversarial attacks by using dropout scheme in the testing phase. For more details, please refer to our [paper](https://arxiv.org/abs/1809.05165) at ICCAD 2018.

Cite our paper using:
```
@inproceedings{wang2018defensive,
  title={Defensive dropout for hardening deep neural networks under adversarial attacks},
  author={Wang, Siyue and Wang, Xiao and Zhao, Pu and Wen, Wujie and Kaeli, David and Chin, Peter and Lin, Xue},
  booktitle={Proceedings of the International Conference on Computer-Aided Design},
  pages={1--8},
  year={2018}
}
```

## Dependencies
This code requires `python 3` and the following packages: `tensorflow`,
`keras`, `numpy`.

This code is tested with `tensorflow 1.15.0`, `keras 2.3.1` and `numpy 1.16.4`.

## Train HRS
`python train_dropout_model.py [options]`

Options:
* `--dataset`: CIFAR or MNIST.
* `--dropout_rate`: dropout rate in training.
* `--train_epoch`: train epochs.
* `--load_pretrain`: path to load pretrain weight.

Outputs:
Trained weights will be saved in `./Model/`.

## Compute Test Accuracy
`python test_acc.py [options]`

Options:
* `--dataset`: CIFAR or MNIST.
* `--train_dropout_rate`: dropout rate in training.
* `--test_dropout_rate`: dropout rate in testing.
* `--pretrain_dir`: dir to load pretrained weights.

Outputs:
Test accuracy of the model will be printed. Note: because
of the randomness of model structure, different runs may result in slightly
different results.

## Defend against Adversarial Attack
`python defend_adversarial_attack.py [options]`

Options:
* `--dataset`: CIFAR or MNIST.
* `--train_dropout_rate`: dropout rate in training.
* `--test_dropout_rate`: dropout rate in testing.
* `--pretrain_dir`: dir to load pretrained weights.
* `--test_examples`: number of test examples. 
* `--attack`: FGSM, PGD or CWPGD. 
* `--epsilon`: the L_inf bound of allowed adversarial perturbations. 
* `--num_steps`: number of steps in generating adversarial examples, not work for FGSM.
* `--step_size`: the step size in generating adversarial examples. Default: `0.1`.

Outputs:
Attack success rate (ASR) and averaged distortion will be printed.

