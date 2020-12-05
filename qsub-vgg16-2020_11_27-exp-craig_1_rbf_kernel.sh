#### START ####
#!/bin/bash
#$ -cwd
# error = Merged with joblog
#$ -o /u/scratch/r/rahulm/capstone/joblogs/joblog.$JOB_ID.$TASK_ID
#$ -j y
## Edit the line below as needed:
#$ -l gpu,P4,h_rt=4:00:00,h_data=4G
## Modify the parallel environment
## and the number of cores as needed:
#$ -pe shared 1
# Email address to notify
#$ -M $USER@mail
# Notify when
#$ -m bea
# Job task array
#$ -t 1-27:1

# echo job info on joblog:
echo "Job $JOB_ID.$SGE_TASK_ID started on:   " `hostname -s`
echo "Job $JOB_ID.$SGE_TASK_ID started on:   " `date `
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
EXP_FOLDER="$SCRATCH/capstone/experiments/vgg16-cifar10-2020_11_27"
EXP_NAME="craig_1_rbf_kernel"
python exp_runner.py \
    --exp_config $EXP_FOLDER/config-exp-$EXP_NAME/config-exp.$SGE_TASK_ID.json \
    --model_checkpoint $EXP_FOLDER/training/checkpoints/checkpoint-epoch_best-model.pth \
    --model_config $EXP_FOLDER/config-model.json \
    --out_folder $EXP_FOLDER/pruning-$EXP_NAME

conda deactivate

# echo job info on joblog:
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `hostname -s`
echo "Job $JOB_ID.$SGE_TASK_ID ended on:   " `date `
echo " "
#### STOP ####
