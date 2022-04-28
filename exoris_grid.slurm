#!/bin/sh
#SBATCH --job-name=exorem
#SBATCH --time=5-00:00:00  
#SBATCH --mail-user=christian.wilkinson@obspm.fr --mail-type=ALL
#SBATCH --clusters=astro_thin
#SBATCH --partition=def
#SBATCH --qos=astro_thin_def_long
#SBATCH --account=cwilkinson
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=24

RUNDIR=/travail/cwilkinson/Travail.${SLURM_JOB_ID}

srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 0 > ./temp/output_d_0.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 1 > ./temp/output_d_1.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 2 > ./temp/output_d_2.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 3 > ./temp/output_d_3.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 4 > ./temp/output_d_4.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 5 > ./temp/output_d_5.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 6 > ./temp/output_d_6.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 7 > ./temp/output_d_7.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 8 > ./temp/output_d_8.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 9 > ./temp/output_d_9.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 10 > ./temp/output_d_10.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 11 > ./temp/output_d_11.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 12 > ./temp/output_d_12.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 13 > ./temp/output_d_13.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 14 > ./temp/output_d_14.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 15 > ./temp/output_d_15.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 16 > ./temp/output_d_16.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 17 > ./temp/output_d_17.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 18 > ./temp/output_d_18.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 19 > ./temp/output_d_19.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 20 > ./temp/output_d_20.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 21 > ./temp/output_d_21.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 22 > ./temp/output_d_22.txt &
srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exoris.py 23 > ./temp/output_d_23.txt &
wait

exit 0
