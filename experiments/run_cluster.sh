#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=./logs/%J.out
#SBATCH --error=./logs/%J.err

#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#only use this if you want to receive mails on you job status
#SBATCH --mail-type=BEGIN,END,FAIL

project_dir=.

# execute train CLI
source "$project_dir"/venv/bin/activate
python ./algorithm/main.py \
    --gpu 1 