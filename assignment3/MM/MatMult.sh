#!/bin/sh
#
#PBS -N MatMult
#PBS -l walltime=00:10:00
#PBS -q 02614
#PBS -l nodes=1:gpus=1

cd $PBS_O_WORKDIR

module load cuda
lscpu
./MatMult 64 125 128
