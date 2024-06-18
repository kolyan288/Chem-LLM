#!/bin/bash
conda activate OpenChem
python /home/kolyan288/Pyprojects/OpenChem/launch.py --nproc_per_node=1 /home/kolyan288/Pyprojects/OpenChem/run.py --config_file="./example_configs/tox21_rnn_config.py" --mode="train_eval"
