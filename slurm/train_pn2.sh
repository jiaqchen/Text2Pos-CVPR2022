#!/bin/bash
#SBATCH --job-name="PN++ classify train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --mem=32G
#SBATCH --time=1:59:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source /usr/stud/kolmet/venv_pyg/bin/activate
srun python3 -m training.pointcloud.pointnet2 "$@"