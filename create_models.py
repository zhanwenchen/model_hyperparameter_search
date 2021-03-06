# create_models.py
# Example: python create_models.py 50 hyperparam_ranges.json
import os
import datetime
import random
import argparse
import json
import warnings
import sys

from lib.utils import save_model_params, ensure_dir


trained_models_folder = 'trained_models'


def choose_hyperparameters_from_file(hyperparameter_ranges_file):
    with open(hyperparameter_ranges_file) as f:
        ranges = json.load(f)

    # Load constants.
    input_size = ranges['input_size']

    batch_norm = random.choice(ranges['batch_norm'])
    use_pooling = random.choice(ranges['use_pooling'])

    conv1_num_kernels = random.choice(list(range(*ranges['conv1_num_kernels'])))
    conv1_dropout = random.uniform(*ranges['conv1_dropout'])

    conv2_num_kernels = random.choice(list(range(*ranges['conv2_num_kernels'])))
    conv2_dropout = random.uniform(*ranges['conv2_dropout'])

    # Randomly choose model hyperparameters from ranges.
    conv1_kernel_size_range = list(range(*ranges['conv1_kernel_size']))
    conv1_stride_range = ranges['conv1_stride']

    pool1_kernel_size_range = ranges['pool1_kernel_size']
    pool1_stride_range = ranges['pool1_stride']

    conv2_kernel_size_range = list(range(*ranges['conv2_kernel_size']))
    conv2_stride_range = ranges['conv2_stride']

    pool2_kernel_size_range = ranges['pool2_kernel_size']
    pool2_stride_range = ranges['pool2_stride']

    # Size-constrained random hyperparameter search
    possible_size_combinations = []
    for conv1_kernel_size in conv1_kernel_size_range:
        for conv1_stride in conv1_stride_range:
            # Satisfy conv1 condition
            if (input_size - conv1_kernel_size) % conv1_stride != 0:
                continue
            conv1_output_size = (conv1_num_kernels, (input_size - conv1_kernel_size) / conv1_stride + 1)

            for pool1_kernel_size in pool1_kernel_size_range:
                for pool1_stride in pool1_stride_range:
                    if (conv1_output_size[1] - pool1_kernel_size) % pool1_stride != 0:
                        continue
                    pool1_output_size = (conv1_num_kernels, (conv1_output_size[1] - pool1_kernel_size) / pool1_stride + 1)

                    for conv2_kernel_size in conv2_kernel_size_range:
                        for conv2_stride in conv2_stride_range:
                            if (pool1_output_size[1] - conv2_kernel_size) % conv2_stride != 0:
                                continue
                            conv2_output_size = (conv2_num_kernels, (pool1_output_size[1] - conv2_kernel_size) / conv2_stride + 1)

                            for pool2_kernel_size in pool2_kernel_size_range:
                                for pool2_stride in pool2_stride_range:
                                    if (conv2_output_size[1] - pool2_kernel_size) % pool2_stride != 0:
                                        continue
                                    pool2_output_size = (conv2_num_kernels, (conv2_output_size[1] - pool2_kernel_size) / pool2_stride + 1)

                                    possible_size_combinations.append((conv1_kernel_size, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride))


    if len(possible_size_combinations) == 0:
        raise ValueError('create_models: no possible combination for pool1 given conv1_output_size[1] = ' + str(conv1_output_size[1]) + '; pool1_kernel_size_ranges = ' + str(pool1_kernel_size_ranges) + '; pool1_stride_ranges = ' + str(pool1_stride_ranges))

    conv1_kernel_size, conv1_stride, pool1_kernel_size, pool1_stride, conv2_kernel_size, conv2_stride, pool2_kernel_size, pool2_stride = random.choice(possible_size_combinations)


    fcs_hidden_size = random.choice(list(range(*ranges['fcs_hidden_size'])))
    fcs_num_hidden_layers = random.choice(list(range(*ranges['fcs_num_hidden_layers'])))
    fcs_dropout = random.uniform(*ranges['fcs_dropout'])


    # Randomly choose training hyperparameters from ranges.
    cost_function = random.choice(ranges['cost_function'])
    optimizer = random.choice(ranges['optimizer'])

    if optimizer == 'SGD':
        momentum = random.uniform(*ranges['momentum'])
        learning_rate = random.uniform(*ranges['learning_rate_sgd'])
    elif optimizer == 'Adam':
        momentum = None
        learning_rate = random.uniform(*ranges['learning_rate_adam'])

    batch_size = random.choice(ranges['batch_size'])
    weight_decay = random.uniform(*ranges['weight_decay'])
    patience = random.choice(ranges['patience'])

    hyperparameters = {
        'input_size': input_size,
        'output_size': ranges['output_size'],

        'batch_norm': batch_norm,

        'use_pooling': use_pooling,
        'pooling_method': ranges['pooling_method'],

        'conv1_kernel_size': conv1_kernel_size,
        'conv1_num_kernels': conv1_num_kernels,
        'conv1_stride': conv1_stride,
        'conv1_dropout': conv1_dropout,

        'pool1_kernel_size': pool1_kernel_size,
        'pool1_stride': pool1_stride,

        'conv2_kernel_size': conv2_kernel_size,
        'conv2_num_kernels': conv2_num_kernels,
        'conv2_stride': conv2_stride,
        'conv2_dropout': conv2_dropout,

        'pool2_kernel_size': pool2_kernel_size,
        'pool2_stride': pool2_stride,

        'fcs_hidden_size': fcs_hidden_size,
        'fcs_num_hidden_layers': fcs_num_hidden_layers,
        'fcs_dropout': fcs_dropout,

        'cost_function': cost_function,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'momentum': momentum,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'patience': patience,
    }

    return hyperparameters


def create_models(num_networks, hyperparameter_ranges_file):
    identifier = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    for count in range(num_networks):
        model_params = choose_hyperparameters_from_file(hyperparameter_ranges_file)

        model_params['train_features'] = os.path.join('data', 'train-images-idx3-ubyte.gz')
        model_params['train_labels'] = os.path.join('data', 'train-labels-idx3-ubyte.gz')
        model_params['valid_features'] = os.path.join('data', 't10k-images-idx3-ubyte.gz')
        model_params['valid_labels'] = os.path.join('data', 't10k-labels-idx3-ubyte.gz')

        model_params['save_dir'] = os.path.join(trained_models_folder, identifier + '_' + str(count+1) + '_created')
        ensure_dir(model_params['save_dir'])
        save_model_params(os.path.join(model_params['save_dir'], 'model_params.txt'), model_params)

    return identifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('num_networks', type=int, help='The number of networks to train.')
    parser.add_argument('hyperparameter_ranges_file', type=str, help='A JSON file to specify the hyperparameter search ranges.')
    args = parser.parse_args()

    num_networks = args.num_networks
    hyperparameter_ranges_file = args.hyperparameter_ranges_file
    return create_models(num_networks, hyperparameter_ranges_file)


if __name__ == '__main__':
    main()
