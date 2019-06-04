import argparse

from model.reader import fer, ck

available_datasets = ['fer', 'ck+']
used_module = None

def set_dataset(name):
    if name not in available_datasets: raise Exception('Unknown dataset name!!')
    global used_module
    if name == 'fer':
        used_module = fer
    elif name == 'ck+':
        used_module = ck


def read_training(limit=-1):
    return used_module.read_training(limit)

def read_testing(limit=-1):
	return used_module.read_testing(limit)

def get_emotions():
	return used_module.emotions