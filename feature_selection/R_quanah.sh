#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N RJOB
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.e$JOB_ID
#$ -q omni
#$ -P quanah
#$ -pe sm 36

#Load the latest version of the R language - compiled using the Intel compilers.
module load intel R

#Allow R to perform some automatic parallelization.
#	MKL_NUM_THREADS - The maximum number of threads you want R to spawn on your behalf.
#	$NSLOTS - This will be replaced by the number of slots you request in yout parallel environment.
#		Example:  -pe sm 36 -> $NSLOTS=36.
export MKL_NUM_THREADS=$NSLOTS

#Run the example R script using the Rscript application.
Rscript Mikaela_example_glmnet.R 

