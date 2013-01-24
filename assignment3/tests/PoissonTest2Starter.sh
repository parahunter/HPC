#!/bin/sh
#
#PBS -N PoissonTest2
#PBS -l walltime=00:10:00
#PBS -q 02614
#PBS -l nodes=1:gpus=1

cd $PBS_O_WORKDIR

module load cuda

python ./PoissonTest2.py
