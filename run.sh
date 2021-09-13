#!/bin/bash

module load anaconda
module load learning/conda-2020.11-py38-gpu
module load ml-toolkit-gpu/pytorch/1.7.1

# python main.py -T 0.01
python snooker_test.py
# python ignore.py
