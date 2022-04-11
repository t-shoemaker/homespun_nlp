#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_performance(model, model_name='', metric=None, size=10):
    chunks = zip(
        np.array_split(model.training_info['iter'], size),
        np.array_split(model.training_info[metric], size)
    )
    graph_data = [(a[0], b[0]) for a, b in chunks]
    x = [i[0] for i in graph_data]
    y = [i[1] for i in graph_data]

    fig = plt.figure(figsize=(15,9))
    plt.plot(x, y)
    plt.xlabel('iterations')
    plt.ylabel(metric)

    plt.savefig(f"{model_name}_{metric}.png", format='png')

def load_vanilla(filepath):
    with open(filepath, 'rb') as fin:
        model = pickle.load(fin)
        if model.backend == 'vanilla':
            return model
        else:
            raise Exception("You must load a vanilla model!")

def save_vanilla(model, filepath):
    if model.backend == 'vanilla':
        with open(filepath, 'wb') as fout:
            pickle.dump(model, fout)
    else:
        raise Exception("You can only use this to save vanilla models!")
