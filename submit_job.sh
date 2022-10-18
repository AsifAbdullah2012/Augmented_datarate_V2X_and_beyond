#!/bin/bash
#
### ADAPT TO YOUR PREFERRED SLURM OPTIONS ###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asif.abdullah.rokoni@hhi.fraunhofer.de
#SBATCH --job-name="NN_loop"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32GB



# include the definition of the LOCAL_JOB_DIR which is autoremoved after each job
source "/etc/slurm/local_job_dir.sh"

### ADAPT TO YOUR DATA DIRECTORY ###
DATA_DIR="/home/fe/rokoni/clean_code_v2/data"
DATA_MNT="/mnt/my_dataset"

# Use input as source code directory if present, otherwise resort to default
if [ $# -eq 0 ]
  then
    ### ADAPT TO YOUR SOURCE CODE DIRECTORY ###
    CODE_DIR="/home/fe/rokoni/clean_code_v2/clean_code"
  else
    CODE_DIR=$1
fi
CODE_MNT="/mnt/project"

# cd ${CODE_DIR}
# python3 test.py $SLURM_ARRAY_TASK_ID

# cd ..

# PETS=("01" "02" "03" "04")

# ANIMAL=${PETS[$SLURM_ARRAY_TASK_ID]}

### ADAPT TO YOUR MAIN SCRIPT ###
singularity run --nv --bind ${DATA_DIR}:${DATA_MNT},${CODE_DIR}:${CODE_MNT} 17_10_22_container.sif /home/fe/rokoni/clean_code_v2/clean_code/main.py
               # --input "${DATA_MNT}/${ANIMAL}.png" --output "${SLURM_ARRAY_JOB_ID}-${ANIMAL}"                      \
               # --grayscale --plot --compression 0.95
