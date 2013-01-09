#!/bin/sh
#PBS -N cpuinfo
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -l walltime=1.00

cd $PBS_O_WORKDIR

lscpu 

# time to say 'Good bye' ;-)
#
exit 0

