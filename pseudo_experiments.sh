#!/bin/bash
#SBATCH --job-name=piaf_expes         # name of job
#SBATCH --partition=gpu_p1            # GPU partition requested
#SBATCH --ntasks=1                    # number of processes (= number of GPUs here)
#SBATCH --gres=gpu:1                  # number of GPUs to reserve
#SBATCH --cpus-per-task=10            # number of cores to reserve (a quarter of the node here)
#SBATCH --hint=nomultithread          # hyperthreading is deactivated
#SBATCH --time=00:30:00               # maximum execution time requested (HH:MM:SS)
#SBATCH --output=piaf_expes%j.out     # name of output file
#SBATCH --error=piaf_expes_errors%j.out      # name of error file (here, in common with the output file)
 
read -s -p "passphrase" passphrase
# cleans out the modules loaded in interactive and inherited by default 
module purge
 
# loading of modules
module load pytorch-gpu/py3/1.4.0
conda activate conseil_etat

 
# move to the project dir
cd $WORK/pseudo_conseil_etat


echo  "Experiment 1: 80_10_10"

gpg -d --batch --passphrase "$passphrase" --output data/80_10_10/train.txt data/80_10_10/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/80_10_10/dev.txt  data/80_10_10/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/80_10_10/test.txt data/80_10_10/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/80_10_10 models/80_10_10
set +x

shred -uvz data/80_10_10/train.txt
shred -uvz data/80_10_10/dev.txt
shred -uvz data/80_10_10/test.txt


echo  "Experiment 2: 160_20_20"

gpg -d --batch --passphrase "$passphrase" --output data/160_20_20/train.txt data/160_20_20/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/160_20_20/dev.txt  data/160_20_20/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/160_20_20/test.txt data/160_20_20/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/160_20_20 models/160_20_20
set +x

shred -uvz data/160_20_20/train.txt
shred -uvz data/160_20_20/dev.txt
shred -uvz data/160_20_20/test.txt

echo  "Experiment 3: 400_50_50"

gpg -d --batch --passphrase "$passphrase" --output data/400_50_50/train.txt data/400_50_50/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/400_50_50/dev.txt  data/400_50_50/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/400_50_50/test.txt data/400_50_50/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/400_50_50 models/400_50_50
set +x

shred -uvz data/400_50_50/train.txt
shred -uvz data/400_50_50/dev.txt
shred -uvz data/400_50_50/test.txt

echo  "Experiment 4: 600_75_75"

gpg -d --batch --passphrase "$passphrase" --output data/600_75_75/train.txt data/600_75_75/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/600_75_75/dev.txt  data/600_75_75/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/600_75_75/test.txt data/600_75_75/test.txt.gpg

srun -n1 python -m src.models.flair_baseline_model data/600_75_75 models/600_75_75
set +x

shred -uvz data/600_75_75/train.txt
shred -uvz data/600_75_75/dev.txt
shred -uvz data/600_75_75/test.txt

echo  "Experiment 5: 800_100_100"
expe=800_100_100

gpg -d --batch --passphrase "$passphrase" --output data/$expe/train.txt data/$expe/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/dev.txt  data/$expe/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/test.txt data/$expe/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/$expe models/$expe
set +x

shred -uvz data/$expe/train.txt
shred -uvz data/$expe/dev.txt
shred -uvz data/$expe/test.txt

echo  "Experiment 6: 1200_150_150"
expe=1200_150_150

gpg -d --batch --passphrase "$passphrase" --output data/$expe/train.txt data/$expe/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/dev.txt  data/$expe/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/test.txt data/$expe/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/$expe models/$expe
set +x

shred -uvz data/$expe/train.txt
shred -uvz data/$expe/dev.txt
shred -uvz data/$expe/test.txt

echo  "Experiment 6: 1600_200_200"
expe=1600_200_200

gpg -d --batch --passphrase "$passphrase" --output data/$expe/train.txt data/$expe/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/dev.txt  data/$expe/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/test.txt data/$expe/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/$expe models/$expe
set +x

shred -uvz data/$expe/train.txt
shred -uvz data/$expe/dev.txt
shred -uvz data/$expe/test.txt

echo  "Experiment 6: 2400_300_300"
expe=2400_300_300

gpg -d --batch --passphrase "$passphrase" --output data/$expe/train.txt data/$expe/train.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/dev.txt  data/$expe/dev.txt.gpg

gpg -d --batch --passphrase "$passphrase" --output data/$expe/test.txt data/$expe/test.txt.gpg

set -x
srun -n1 python -m src.models.flair_baseline_model data/$expe models/$expe
set +x

shred -uvz data/$expe/train.txt
shred -uvz data/$expe/dev.txt
shred -uvz data/$expe/test.txt


echo "All done!"    
scancel -u uzf35gs
