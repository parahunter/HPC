#!/bin/sh
#
#PBS -N TestJacobi_12_12_32
#PBS -l walltime=00:10:00
#PBS -q 02614
#PBS -l nodes=1:ppn=8:gpus=1

cd $PBS_O_WORKDIR

module load cuda

ONP_NUM_THREADS=8 ./Jacobi 2048 100 16
