#!/bin/bash
#SBATCH --job-name=run_maverick_6
#SBATCH --output=run_maverick/%j.out
#SBATCH --error=run_maverick/%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --exclude=n-101

# Set environment variables
export PYTHONPATH=/home/morg/students/gottesman3/knowledge-analysis-suite/maverick-coref
export HOME=/home/morg/students/gottesman3

echo "GPU 6"

# Run the Python script with the specified arguments
python /home/morg/students/gottesman3/knowledge-analysis-suite/maverick-coref/run_maverick.py \
  --gpu_id 6 \
  --n_gpus 8 \
  --max_memory_gb 75
