#!/bin/sh
#SBATCH --job-name=exo_evol
#SBATCH --time=0-23:00:00  
#SBATCH --mail-user=christian.wilkinson@obspm.fr --mail-type=ALL
#SBATCH --clusters=astro_thin
#SBATCH --partition=def
#SBATCH --qos=astro_thin_def_long
#SBATCH --account=cwilkinson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24


RUNDIR=/travail/cwilkinson/Travail/exo_evol.${SLURM_JOB_ID}

echo Merging files
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/merge_files.py > ./temp/merge_files.txt
echo Completed file merge

srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 0 > ./temp/exo_evol_0.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 1 > ./temp/exo_evol_1.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 2 > ./temp/exo_evol_2.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 3 > ./temp/exo_evol_3.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 4 > ./temp/exo_evol_4.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 5 > ./temp/exo_evol_5.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 6 > ./temp/exo_evol_6.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 7 > ./temp/exo_evol_7.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 8 > ./temp/exo_evol_8.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 9 > ./temp/exo_evol_9.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 10 > ./temp/exo_evol_10.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 11 > ./temp/exo_evol_11.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 12 > ./temp/exo_evol_12.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 13 > ./temp/exo_evol_13.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 14 > ./temp/exo_evol_14.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 15 > ./temp/exo_evol_15.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 16 > ./temp/exo_evol_16.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 17 > ./temp/exo_evol_17.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 18 > ./temp/exo_evol_18.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 19 > ./temp/exo_evol_19.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 20 > ./temp/exo_evol_20.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 21 > ./temp/exo_evol_21.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 22 > ./temp/exo_evol_22.txt
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/exo_evol/link_grids.py 23 > ./temp/exo_evol_23.txt


wait

exit 0