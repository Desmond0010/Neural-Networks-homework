"""
Author: Sophia Sanborn
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas
"""

"""
Step 1: Define layer arguments

- Define the arguments for each layer in an attribute dictionary (AttrDict).
- An attribute dictionary is exactly like a dictionary, except you can access the values as attributes rather than keys...for cleaner code :)
- See layers.py for the arguments expected by each layer type.
"""

from neural_networks.utils.data_structures import AttrDict


hidden_layer_size=10
lr=1e-2
model_name = "two_layer_NN"

layer_1 = AttrDict(
    {
        "name": "fully_connected",
        "activation": "relu",
        "weight_init": "xavier_uniform",
        "n_out": hidden_layer_size,
    }
)

layer_out = AttrDict(
    {
        "name": "fully_connected",
        "activation": "softmax",
        "weight_init": "xavier_uniform",
        "n_out": 3
        # n_out is not defined for last layer. This will be set by the dataset.
    }
)

"""
Step 2: Collect layer argument dictionaries into a list.

- This defines the order of layers in the network.
"""

layer_args = [layer_1, layer_out]

"""
Step 3: Define model, data, and logger arguments

- The list of layer_args is passed to the model initializer.
"""

optimizer_args = AttrDict(
    {
        "name": "SGD",
        "lr": lr,
        "lr_scheduler":"exponential", # YOUR CODE HERE,alternative is "constant"
        "lr_decay": 0.99,# YOUR CODE HERE,
        "stage_length": 1000,# YOUR CODE HERE,
        "staircase": False,# YOUR CODE HERE,
        "clip_norm": None,# YOUR CODE HERE,
        "momentum": 0# YOUR CODE HERE,
    }
)



model_args = AttrDict(
    {
        "name":model_name,
        "loss": "cross_entropy",
        "layer_args": layer_args,
        "optimizer_args": optimizer_args,
        "seed": 2
    }
)

data_args = AttrDict(
    {
        "name": "iris",# YOUR CODE HERE, name of dataset, e.g. "iris"
        "batch_size": 50# YOUR CODE HERE,
    }
)

log_args = AttrDict(
    {"save": True, "plot": True, "save_dir": "experiments/",}
)

"""
Step 4: Set random seed

Warning! Random seed must be set before importing other modules.
"""

import numpy as np

np.random.seed(model_args.seed)

"""
Step 5: Define model name for saving
"""


"""
Step 6: Initialize logger, model, and dataset

- model_name, model_args, and data_args are passed to the logger for saving
- The logger is passed to the model.
"""

from neural_networks.models import initialize_model
from neural_networks.datasets import initialize_dataset
from neural_networks.logs import Logger


logger = Logger(
    model_name=model_name,
    model_args=model_args,
    data_args=data_args,
    save=log_args.save,
    plot=log_args.plot,
    save_dir=log_args.save_dir,
)


model = initialize_model(
    name=model_args.name,
    loss=model_args.loss,
    layer_args=model_args.layer_args,
    optimizer_args=model_args.optimizer_args,
    logger=logger,
)


dataset = initialize_dataset(
    name=data_args.name,
    batch_size=data_args.batch_size,
)


"""
Step 7: Train model!
"""

epochs = 100
#epochs=4
print(
    "Training {} neural network on {} with {} for {} epochs...".format(
        model_args.name, data_args.name, optimizer_args.name, epochs
    )
)

print("Optimizer:")
print(optimizer_args)

model.train(dataset, epochs=epochs)
model.test(dataset) # For Higgs, call test_kaggle() to generate the Kaggle file.
