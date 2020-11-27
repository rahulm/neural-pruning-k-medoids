#### START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o joblog.$JOB_ID
#$ -j y
## Edit the line below as needed:
#$ -l gpu,P4,h_rt=8:00:00,h_data=4G,h_vmem=4G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea

# TODO: Need to update the h_rt and h_data as needed to run experiments.

# echo job info on joblog:
echo "Job $JOB_ID started on:   " `hostname -s`
echo "Job $JOB_ID started on:   " `date `
echo " "


# Set up environment
if command -v conda &> /dev/null; then conda deactivate; fi
source /u/local/Modules/default/init/modules.sh
module load python/anaconda3
. "/u/local/apps/anaconda3/etc/profile.d/conda.sh"

#conda activate pytorch-1.5.0-cpu
#conda activate pytorch-1.3.1-gpu
conda activate capstone

# Run training
OUT_FOLDER="$SCRATCH/capstone/experiments/vgg16-cifar10-2020_11_26"
python exp_runner.py \
    --exp_config $OUT_FOLDER/config-exp-craig_1.json \
    --model_checkpoint $OUT_FOLDER/training/checkpoints/checkpoint-epoch_best-model.pth \
    --model_config $OUT_FOLDER/config-model.json \
    --out_folder $OUT_FOLDER/pruning-craig_1

conda deactivate
