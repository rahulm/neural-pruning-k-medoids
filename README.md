# ucla-ms-capstone
Capstone project for UCLA Masters program in Computer Science, 2020.


## Brief overview
- [pruner.py](pruner.py) is used to prune a network (currently only LeNet-300-100) based on CRAIG or Mussay. For CRAIG, we can define more similarity metrics in the SimilarityMetrics class as we experiment. We pass a JSON file as a PruneConfig to denote what we want to do.
- [eval_model.py](eval_model.py) is to evaluate a given model (currently only LeNet-300-100) on a dataset (currently only MNIST). Returns both accuracy and average loss.
- [train_algo_1.py](train_algo_1.py) is used to train a model, based on a train config JSON file.

## Examples
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
    },
    "original_model_path": "experiments/lenet_300_100-finetuned/training/checkpoints/checkpoint-epoch_40-model.pth"
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
    },
    "original_model_path": "experiments/lenet_300_100-finetuned/training/checkpoints/checkpoint-epoch_40-model.pth"
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

