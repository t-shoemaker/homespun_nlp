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

    if metric == 'perp':
        plt.ticklabel_format(style='plain')
        plt.yscale('log')

    plt.savefig(f"{model_name}_{metric}.png", format='png')

def load_rnn(filepath):
    with open(filepath, 'rb') as fin:
        return pickle.load(fin)

def save_rnn(model, filepath):
    with open(filepath, 'wb') as fout:
        pickle.dump(model, fout)
