#!/bin/bash
#SBATCH --job-name=piaf_expes         # name of job
#SBATCH --partition=gpu_p1            # GPU partition requested
#SBATCH --ntasks=1                    # number of processes (= number of GPUs here)
#SBATCH --gres=gpu:1                  # number of GPUs to reserve
#SBATCH --cpus-per-task=10            # number of cores to reserve (a quarter of the node here)
#SBATCH --hint=nomultithread          # hyperthreading is deactivated
#SBATCH --time=13:00:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=piaf_expes%j.out     # name of output file
#SBATCH --error=piaf_expes_errors%j.out      # name of error file (here, in common with the output file)
 
read -s -p "passphrase" passphrase
# cleans out the modules loaded in interactive and inherited by default 
module purge
 
# loading of modules
module load pytorch-gpu/py3/1.4.0

# echo of launched commands
set -x
 
# move to the project dir
cd $WORK/piafing_around


echo  "Experiment 1: 80_10_10"

gpg -c --batch --passphrase $passphrase data/80_10_10/train.txt.gpg
gpg -c --batch --passphrase $passphrase data/80_10_10/dev.txt.gpg
gpg -c --batch --passphrase $passphrase data/80_10_10/test.txt.gpg

srun -n1 python f srun -n1 python -m src.models.flair_baseline_model data/80_10_10 models/80_10_10

shred -uvz data/80_10_10/train.txt
shred -uvz data/80_10_10/dev.txt
shred -uvz data/80_10_10/test.txt

echo  "Experiment 2: Squad-FR-train"

echo "All done!"    
scancel -u uzf35gs