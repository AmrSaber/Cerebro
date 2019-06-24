import argparse

from Cerebro.model.reader import fer, ck, rafd, combined

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
    elif name == 'combined':
        used_module = combined

def read_training(limit=-1):
    return used_module.read_training(limit)

def read_testing(limit=-1):
	return used_module.read_testing(limit)

def get_emotions():
	return used_module.emotions

def get_reduced_emotions():
    return used_module.reduced_emotions