#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00

#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#only use this if you want to receive mails on you job status
#SBATCH --mail-type=BEGIN,END,FAIL

project_dir=.

# execute train CLI
source "$project_dir"/venv/bin/activate
python ./algorithm/main.py \
    --root "$project_dir"/data/data \
    --gpu 1 