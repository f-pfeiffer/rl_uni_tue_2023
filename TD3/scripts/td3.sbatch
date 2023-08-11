#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=TD3

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=4G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=23:59:57
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"

#SBATCH --error=tcml_logs/td3.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=tcml_logs/td3.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=f.pfeiffer@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####

# In this case, none.

####
#c) Execute your  code in a specific singularity container
####

##################################################################################################
# TODO: update the following paths with your own!
singularity exec --nv /home/stud46/rl_lecture_wandb.simg python3 /home/stud46/felix/train_td3.py \
    --steps 2000000 \
    --alpha 0.001 \
    --beta 0.001 \
    --gamma 0.99 \
    --tau 0.005 \
    --max_buffer_size 1000000 \
    --batch_size 256 \
    --update_actor_interval 2 \
    --warmup 1000 \
    --noise 0.1 \
    --save_path runs/td3/ \
    --device cuda \
    --save_freq 50000 \
    --eval_freq 50000 \
    --eval_eps 500 \
    --continue_training_step 250000 \
    --path_pretrained runs/td3/84dxy366/models/
##################################################################################################
