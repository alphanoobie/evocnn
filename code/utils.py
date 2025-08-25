import numpy as np
import os
import pickle
from time import gmtime, strftime
from individual import *

# Helpers for consistent, project-relative storage
def _project_root():
    # Directory containing this utils.py file
    return os.path.dirname(os.path.abspath(__file__))

def _data_dir():
    return _project_root()

def get_data_path():
    # pops.dat saved next to code files (same as previous behavior but robust)
    return os.path.join(_data_dir(), 'pops.dat')

def _ensure_parent_dir(path):
    # Create parent directory if missing
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def save_populations(gen_no, pops):
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    path = get_data_path()
    _ensure_parent_dir(path)
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)

def load_population():
    path = get_data_path()
    with open(path, 'rb') as file_handler:
        data = pickle.load(file_handler)
    return data['gen_no'], data['pops'], data['create_time']

def save_offspring(gen_no, pops):
    data = {'gen_no': gen_no, 'pops': pops, 'create_time': strftime("%Y-%m-%d %H:%M:%S", gmtime())}
    # Save under project_root/offsprings_data/gen_{n}.dat (not cwd)
    dirpath = os.path.join(_data_dir(), 'offsprings_data')
    os.makedirs(dirpath, exist_ok=True)  # ensure directory exists
    path = os.path.join(dirpath, f'gen_{gen_no}.dat')
    with open(path, 'wb') as file_handler:
        pickle.dump(data, file_handler)

def load_save_log_data():
    # Legacy absolute path kept for reference; adjust or remove as needed
    file_name = '/am/lido/home/yanan/eclipse-workspace/Ver3/pops.dat'
    with open(file_name, 'br') as file_h:
        data = pickle.load(file_h)
        print(data)
        pops = data['pops'].pops
        for i in range(len(pops)):
            print(pops[i])

def save_append_individual(indi, file_path):
    _ensure_parent_dir(file_path)
    with open(file_path, 'a') as myfile:
        myfile.write(indi)
        myfile.write("\n")

def randint(low, high):
    return np.random.randint(low, high)

def rand():
    return np.random.random()

def flip(f):
    return rand() <= f

if __name__ == '__main__':
    load_save_log_data()
