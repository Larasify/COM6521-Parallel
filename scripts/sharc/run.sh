#!/bin/bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
# Request 1 gigabytes of real memory (mem) per core
#$ -l rmem=1G
# Request 4 cores in an OpenMP environment (this also causes 4 GPUs to be requested!)
#$ -pe openmp 4
# Name the job
#$ -N com4521_run
# Request 10 minutes of time (This should be more than enough for your assignment)
#$ -l h_rt=00:10:00
# To enable email notification, update the email address
#$ -M me@somedomain.com
# Email notifications if the job begins/ends/aborts (remove the characters as desired)
#$ -m bea

# Load Modules
module load libs/CUDA/11.6.0/binary
module load dev/gcc/8.2

# Run the compiled program

./bin/release/Particles CUDA 100 512x512 output_image.png --bench