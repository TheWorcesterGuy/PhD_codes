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

CPU_counter=0
for (( c=0; c<=100; c++ ))
do  
    echo $CPU_counter
    srun -u --exclusive --ntasks 1 python3 /travail/cwilkinson/Travail/grille_exorem.py $c > ./temp/output_exorem_d_$c.txt &
    let "CPU_counter+=1" 
    
    if [[ "$CPU_counter" -eq 24 ]]; then
        echo Waiting for processes
        CPU_counter=0
        wait
    fi

done
wait

exit 0