#!/bin/bash

if command -v conda &> /dev/null; then conda deactivate; fi
source /u/local/Modules/default/init/modules.sh
module load python/anaconda3
. "/u/local/apps/anaconda3/etc/profile.d/conda.sh"

#conda activate pytorch-1.5.0-cpu
#conda activate pytorch-1.3.1-gpu
conda activate capstone

# Run training
OUT_FOLDER="$SCRATCH/capstone/experiments/vgg16-cifar10-2020_11_26"
python train_algo_1.py \
    --train_config $OUT_FOLDER/config-train.json \
    --model_config $OUT_FOLDER/config-model.json \
    --out_folder $OUT_FOLDER/training \
    --save_interval 0
    --save_best_checkpoint

conda deactivate
