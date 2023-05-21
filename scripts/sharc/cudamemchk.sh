#!/bin/bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
# Request 1 gigabytes of real memory (mem) per core
#$ -l rmem=1G
# Request 1 core in an OpenMP environment
#$ -pe openmp 1
# Name the job
#$ -N com4521_cudamemchk
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

compute-sanitizer --print-limit 1 "bin/debug/Particles" CUDA 100 512x512 output_image.png --bench