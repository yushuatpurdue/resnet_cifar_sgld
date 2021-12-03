#!/bin/bash

module load anaconda
module load learning/conda-2020.11-py38-gpu
module load ml-toolkit-gpu/pytorch/1.7.1

python main.py -T 0.01 -epochs 1000 -method 'sghmc' -nchains 1
# python main.py -T 0.05 -epochs 1000 -method 're' -nchains 2
# python snooker_test.py
# python ignore.py
