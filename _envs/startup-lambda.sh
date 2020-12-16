#!/usr/bin/env bash

yes | pip install torch==1.7.1 torchvision

cd ~
mkdir capstone
cd capstone
git clone https://github.com/rahulm/ucla-ms-capstone.git
cd ucla-ms-capstone

python data_downloader.py --data cifar10

# Experiments specific
mkdir experiments
cd experiments
mkdir vgg16-cifar10-2020_12_11
