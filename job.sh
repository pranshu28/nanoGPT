#!/bin/bash
#SBATCH --job-name=owt_gpt
#SBATCH --output=/network/scratch/p/pranshu.malviya/projects/Optimization/nanoGPT/logs/%x-%j_output.txt
#SBATCH --error=/network/scratch/p/pranshu.malviya/projects/Optimization/nanoGPT/logs/%x-%j_error.txt
#SBATCH --time=3:00:00
#SBATCH --partition=short-unkillable,long    # ask for high-priority job ## short-unkillable,
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # crucial - only 1 task per node!
#SBATCH --cpus-per-task=24              # number of cpus per node
#SBATCH --gpus-per-task=4               # number of gpus per node
#SBATCH --mem=128G               # memory per gpu
#SBATCH --signal=TERM@60                # SIGTERM 60s prior to the allocation's end
#SBATCH --switches=1@01:00:00           # number of leaf switches (InfiniBand Island) with time limit for the constraint
#SBATCH --constraint=80gb               # constraints


# HuggingFace cache folders
export HF_HOME=$SCRATCH/projects/Optimization/nanoGPT/huggingface
export HF_DATASETS_CACHE=$SCRATCH/projects/Optimization/nanoGPT/huggingface/datasets


# Load modules and python environment
module load gcc
module load miniconda
module load cuda/12.1.1
conda activate pla10

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# python data/openwebtext/prepare.py
# torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

rm -r ./confs/$SLURM_JOB_ID
mkdir -p ./confs/$SLURM_JOB_ID
conf_name="./confs/$SLURM_JOB_ID/config.conf"

init_optimizer=$1
init_lr=$2
init_steps=$3
gpus=$4

echo "0 torchrun --standalone --nproc_per_node=$4 train.py \
    config/train_gpt2.py \
    --init_optimizer=$init_optimizer \
    --init_lr=$init_lr \
    --init_steps=$init_steps \
    --init_from='resume' \
    --out_dir='out/$init_optimizer-$init_lr-$init_steps' \
    " >> $conf_name
srun -l --multi-prog $conf_name
