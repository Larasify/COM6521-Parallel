#!/bin/bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
# Request 8 gigabytes of real memory (mem) per core, 1/16 the DGX1's RAM
#$ -l rmem=8G
# Request 1 core in an OpenMP environment
#$ -pe openmp 1
# Name the job
#$ -N com4521_profile
# Request 10 minutes of time (This should be more than enough for your assignment)
#$ -l h_rt=00:10:00
# To enable email notification, update the email address
#$ -M me@somedomain.com
# Email notifications if the job begins/ends/aborts (remove the characters as desired)
#$ -m bea

# Load Modules
module load libs/CUDA/11.6.0/binary
module load dev/gcc/8.2

# Run the compiled program through the two profilers

# Generate timeline
nvprof -o timeline.nvvp "./bin/release/Particles" CUDA 100 512x512
# Generate analysis metrics
nvprof --analysis-metrics -o analysis.nvprof "./bin/release/Particles" CUDA 100 512x512

## Not required, visual profiler has a timeline
#Nsight Systems
#nsys --version
#nsys profile -o <filename> <executable> <args>
#nsys profile -o "nsys-out" "bin/release/Particles" CUDA 100 512x512

## Note Nsight Compute Does not work on Pascal and below GPUs!!
#Nsight Compute
#ncu --version
#ncu --set full -o <filename> <executable> <args>
#ncu --set full -o "ncu-out" "bin/release/Particles" CUDA 100 512x512
