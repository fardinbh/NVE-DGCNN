#!/bin/bash
#SBATCH --gres=gpu:4        # request GPU "generic resource"
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=24   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=123G        # memory per node
#SBATCH --begin=2020-02-11T01:30:00     # time (DD-HH:MM)
#SBATCH --time=0-24:00      # time (DD-HH:MM)
#SBATCH --output=Train2-New2%j.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6
source ~/tensorflow/bin/activate
module load cuda
python ./train-eval.py