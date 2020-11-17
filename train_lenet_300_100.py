"""
This script trains a model on the MNIST dataset.
This is not necessariliy generalizable, hyperparameters etc need to be manually
changed in the code.
Specifically, this trains the LeNet-300-100 model.
TODO: Make this generalizable, use a model_config + a more generic FC model.
"""


import os
from datetime import datetime
from typing import Text

import torch
import torch.nn.functional as F
import torchvision

import eval_model
from models.fc_2 import Model
from utils import logging_utils, train_utils

# TODO: Change this per experiment.
experiment_id: Text = "test-1"

data_folder_path: Text = os.path.join("data", "pytorch")
experiment_folder_path: Text = os.path.join("experiments", experiment_id)

logging_utils.setup_logging(
    os.path.join(
        experiment_folder_path,
        "log-{}.txt".format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")),
    )
)
logger = logging_utils.get_logger(name=__name__)

num_epochs: int = 40  # 14
batch_size_train: int = 128  # NOTE: may need to change this.
batch_size_test: int = 1000
learning_rate: float = 0.01  # 1.0
momentum: float = 0.9  # 0.5
weight_decay: float = 0.0001

lr_step_size: int = 10
gamma: float = 0.7

random_seed: int = 1234
log_interval: int = 100
device: Text = "cuda"  # or cpu

torch_device = torch.device(device)
torch.manual_seed(random_seed)

train_loss_batches: train_utils.StatCounter = train_utils.StatCounter()
train_loss_epochs: train_utils.StatCounter = train_utils.StatCounter()
test_loss_epochs: train_utils.StatCounter = train_utils.StatCounter()
test_acc_epochs: train_utils.StatCounter = train_utils.StatCounter()

# NOTE: This is the LeNet-300-100 configuration, test different ones later.
model: Model = Model(
    input_shape=(28, 28), layer_1_dim=300, layer_2_dim=100, output_dim=10
)
model.to(device=torch_device)

# Just using basic Stochastic Gradient Descent.
# TODO: Add weigh decay? May not be necesssary for this task
optimizer = torch.optim.SGD(
    params=model.parameters(),
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay,
)
# optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=lr_step_size, gamma=gamma
)


# Get data.
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_folder_path, train=True, download=True, transform=data_transform
    ),
    batch_size=batch_size_train,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        data_folder_path, train=False, download=True, transform=data_transform
    ),
    batch_size=batch_size_test,
    shuffle=True,
)


def train(epoch: int) -> None:
    model.train()
    curr_loss: float = 0.0
    total_data_count: int = 0
    for batch_ind, (data, target) in enumerate(train_loader):
        total_data_count += len(data)

        data, target = data.to(torch_device), target.to(torch_device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        curr_loss = loss.item()
        train_loss_batches.add(curr_loss)

        if batch_ind % log_interval == 0:
            logger.info(
                "Train || Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    total_data_count,
                    len(train_loader.dataset),
                    100.0 * total_data_count / len(train_loader.dataset),
                    curr_loss,
                )
            )

    train_loss_epochs.add(curr_loss)


checkpoints_folder_path: Text = os.path.join(
    experiment_folder_path, "checkpoints"
)
if not os.path.exists(checkpoints_folder_path):
    os.makedirs(checkpoints_folder_path)

try:
    for epoch in range(1, num_epochs + 1):
        train(epoch)
        test_acc, test_loss = eval_model.evaluate_model(
            model=model, dataloader=test_loader, torchdevice=torch_device
        )
        test_acc_epochs.add(test_acc)
        test_loss_epochs.add(test_loss)
        scheduler.step()

        # Save model checkpoint.
        torch.save(
            model,
            os.path.join(
                checkpoints_folder_path,
                "checkpoint-epoch_{}-model.pth".format(epoch),
            ),
        )
        torch.save(
            model.state_dict(),
            os.path.join(
                checkpoints_folder_path,
                "checkpoint-epoch_{}-weight_only.pth".format(epoch),
            ),
        )
finally:
    # Save losses.
    loss_folder_path: Text = os.path.join(experiment_folder_path, "loss")
    train_loss_batches.save(
        folder_path=loss_folder_path,
        file_prefix="train_loss_batches",
        xlabel="batch",
        ylabel="loss",
        title_prefix="train_loss_batches",
    )
    train_loss_epochs.save(
        folder_path=loss_folder_path,
        file_prefix="train_loss_epochs",
        xlabel="epoch",
        ylabel="loss",
        title_prefix="train_loss_epochs",
    )
    test_loss_epochs.save(
        folder_path=loss_folder_path,
        file_prefix="test_loss_epochs",
        xlabel="epoch",
        ylabel="loss",
        title_prefix="test_loss_epochs",
    )
    test_acc_epochs.save(
        folder_path=loss_folder_path,
        file_prefix="test_accuracy_epochs",
        xlabel="epoch",
        ylabel="accuracy",
        title_prefix="test_accuracy_epochs",
    )
