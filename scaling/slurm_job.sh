#!/bin/bash

#SBATCH -J input_data_scaling
#SBATCH -D /dss/dsshome1/01/di97rov/ROOT_project/scaling
#SBATCH -o /dss/dsshome1/01/di97rov/ROOT_project/scaling/logs/root.out
#SBATCH -e /dss/dsshome1/01/di97rov/ROOT_project/scaling/logs/root.err
#SBATCH --mail-user="christina.krause@stud-mail.uni-wuerzburg.de"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --cluster=hpda2
#SBATCH --partition="hpda2_compute"
#SBATCH --ntasks=1
#SBATCH --mem=128gb
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --time=20:00:00

module load slurm_setup

/dss/dsshome1/01/di97rov/.local/share/mamba/envs/ski/bin/python /dss/dsshome1/01/di97rov/ROOT_project/scaling/generate_input_data_inference.py
