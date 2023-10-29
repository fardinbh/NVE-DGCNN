#!/bin/bash
#SBATCH --gres=gpu:2        # request GPU "generic resource"
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=4   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20G        # memory per node
#SBATCH --begin=2019-03-22T20:30:00     # time (DD-HH:MM)
#SBATCH --time=0-08:50      # time (DD-HH:MM)
#SBATCH --output=batch_inference3-%j.out  # %N for node name, %j for jobID

module load cuda cudnn python/.6
source ~/tensorflow/bin/activate
module load cuda
python ./batch_inference.py