import argparse

from model.reader import fer, ck, rafd

# TODO: implement combined dataset
available_datasets = ['fer', 'ck+', 'rafd', 'combined']
used_module = None

def set_dataset(name):
    global used_module
    if name not in available_datasets: raise Exception('Unknown dataset name!!')
    
    if name == 'fer':
        used_module = fer
    elif name == 'ck+':
        used_module = ck
    elif name == 'rafd':
        used_module = rafd


def read_training(limit=-1):
    return used_module.read_training(limit)

def read_testing(limit=-1):
	return used_module.read_testing(limit)

def get_emotions():
	return used_module.emotions