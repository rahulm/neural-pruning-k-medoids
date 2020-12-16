# neural-pruning-k-medoids
Neural Pruning with K-Medoids, built on the [CRAIG](http://proceedings.mlr.press/v119/mirzasoleiman20a.html) [algorithm](https://github.com/baharanm/craig).

Paper to be included.


## Brief overview
- [pruner.py](pruner.py) is used to prune a network based on CRAIG or Mussay. For CRAIG, we can define more similarity metrics in the SimilarityMetrics class as we experiment. We pass a JSON file as a PruneConfig to denote what we want to do.
- [eval_model.py](eval_model.py) is to evaluate a given model on a dataset. Returns both accuracy and average loss.
- [train_algo_1.py](train_algo_1.py) is used to train a model, based on a train config JSON file.
- [exp_runner.py](exp_runner.py) is used to perform pruning+finetuning experiments based on an experiment config JSON file.

## Example configs
Note that some fields are ignored in certain scripts.
### Model config
```json
{
    "model_architecture": "fc_classifier",
    "model_params": {
        "input_shape": [
            28,
            28
        ],
        "layers": [
            300,
            100
        ],
        "output_dim": 10
    }
}
```

### Prune config

#### CRAIG
```json
{
    "config_id": "config-prune-lenet_300_100-v1",
    "prune_type": "craig",
    "prune_params": {
        "prune_percent_per_layer": 0.3,
        "similarity_metric": "weights_covariance"
    }
}
```

#### Mussay
```json
{
    "prune_type": "mussay",
    "prune_params": {
        "prune_percent_per_layer": 0.3,
        "upper_bound": 1,
        "compression_type": "Coreset"
    }
}
```

### Train config
```json
{
    "algo_name": "train_algo_1",
    "dataset_name": "mnist",
    "num_epochs": 40,
    "batch_size_train": 128,
    "batch_size_test": 1024,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "lr_step_size": 10,
    "gamma": 0.7,
    "random_seed": 1234
}
```

### Experiment config
```json
{
    "prune_type": "craig",
    "model_input_shape": [
        32,
        32,
        3
    ],
    "data_transform_name": "cifar10",
    "prune_params": {
        "layer_params": {
            "all": {
                "prune_type": [
                    "craig"
                ],
                "similarity_metric": [
                    "euclidean_distance",
                    {
                        "name": "rbf_kernel",
                        "gamma": "f^-1"
                    },
                    {
                        "name": "rbf_kernel",
                        "gamma": "sqrt(f)^-1"
                    },
                    "l1_norm",
                    "cosine_similarity"
                ],
                "prune_percent_per_layer": [
                    0.25,
                    0.5,
                    0.75
                ]
            }
        }
    },
    "finetuning_train_config": {
        "algo_name": "train_algo_1",
        "dataset_name": "cifar10",
        "num_epochs": 50,
        "batch_size_train": 128,
        "batch_size_test": 128,
        "learning_rate": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "lr_step_size": 40,
        "gamma": 0.7,
        "random_seed": 1234
    },
    "evaluation_dataset_name": "cifar10",
    "evaluation_dataset_batch_size": 128,
    "evaluation_epochs": [
        0,
        1,
        2,
        3,
        -1,
        "best"
    ],
    "cuda_model_max_mb": 2200,
    "cuda_max_percent_mem_usage": 0.7
}
```

# Acknowledgements
Completed as a capstone project for the UCLA Masters program in Computer Science, Fall 2020.

Faculty Advisor: [Baharan Mirzasoleiman](https://web.cs.ucla.edu/~baharan/)

Committee Member: [Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/)

Committee Member: [Lin Yang](http://drlinyang.net/)

