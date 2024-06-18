import subprocess
import os

conf1 = r"./example_configs/logp_gcnn_config.py"
os.chdir('/home/kolyan288/Pyprojects/OpenChem_local')
result3 = subprocess.run(f'python3 launch.py --nproc_per_node=1 run.py --config_file="./example_configs/logp_gcnn_config.py" --mode="train_eval"', shell = True)




