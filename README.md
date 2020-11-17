# ucla-ms-capstone
Capstone project for UCLA Masters program in Computer Science, 2020.


## Brief overview
- craig_pruner.py is used to prune a network (currently only LeNet-300-100) based on CRAIG. We can define more similarity metrics in the SimilarityMetrics class as we experiment. We pass a JSON file as a PruneConfig to denote what we want to do.
- eval_model.py is to evaluate a given model (currently only LeNet-300-100) on a dataset (currently only MNIST). Returns both accuracy and average loss.
- train_lenet_300_100.py is used to train a model on MNIST. Need to make a more generalizable version for different models.

## Examples
Note that some fields are ignored in certain scripts.
### Model config
```json
{
    "model_architecture": "fc_2",
    "model_params": {
        "input_shape": [
            28,
            28
        ],
        "layer_1_dim": 300,
        "layer_2_dim": 100,
        "output_dim": 10
    },
    "model_path": "experiments/test-1/checkpoints/checkpoint-epoch_20-model.pth"
}
```

### Prune config
```json
{
    "config_id": "test_prune_config-lenet_300_100",
    "model_config": {
        "model_architecture": "fc_2",
        "model_params": {
            "input_shape": [
                28,
                28
            ],
            "layer_1_dim": 300,
            "layer_2_dim": 100,
            "output_dim": 10
        },
        "model_folder": "experiments/test-1/original"
    },
    "prune_type": "craig",
    "prune_params": {
        "prune_percent_per_layer": 0.2,
        "similarity_metric": "weights_covariance"
    },
    "pruned_model_out_folder": "experiments/test-1/pruned"
}
```