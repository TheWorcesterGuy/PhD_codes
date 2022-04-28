#!/bin/sh
#SBATCH --job-name=exo_evol
#SBATCH --time=0-12:00:00  
#SBATCH --mail-user=christian.wilkinson@obspm.fr --mail-type=ALL
#SBATCH --clusters=astro_thin
#SBATCH --partition=def
#SBATCH --qos=astro_thin_def_long
#SBATCH --account=cwilkinson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

python3 -u /travail/cwilkinson/Travail/exo_evol/exo_evol.py > output.txt

exit 0