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
OUT_FOLDER="$SCRATCH/capstone/experiments/vgg16-cifar10-2020_11_27"
python train_algo_1.py \
    --train_config $OUT_FOLDER/config-train.json \
    --model_config $OUT_FOLDER/config-model.json \
    --out_folder $OUT_FOLDER/training \
    --save_interval 0 \
    --save_best_checkpoint

conda deactivate

# echo job info on joblog:
echo "Job $JOB_ID ended on:   " `hostname -s`
echo "Job $JOB_ID ended on:   " `date `
echo " "
#### STOP ####
