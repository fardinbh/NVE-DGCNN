#!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=12   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=123G        # memory per node
#SBATCH --time=0-10:00      # time (DD-HH:MM)
#SBATCH --output=collect-%j.out  # %N for node name, %j for jobID

module load cuda cudnn python/3.6
source ~/tensorflow/bin/activate
module load cuda
python ./collect_indoor3d_data.py