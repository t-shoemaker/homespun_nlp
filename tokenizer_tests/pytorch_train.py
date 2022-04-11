#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import torch
from charmodel.pytorch import Model
from charmodel.utils import plot_performance

def main(args):
    model = Model(args.training_data)
    model.train()

    for metric in ['loss', 'perp']:
        plot_performance(model, model_name=args.model_name, metric=metric)

    model.save_training_info(args.model_name + '_info.json')
    
    samples = []
    for c in ['The', 'A', 'I', 'One']:
        samp = model.sample(200, prime=c)
        samples.append(samp)

    with open(args.model_name + '.txt', 'a') as fout:
        for samp in samples:
            fout.write(samp + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--training_data',
        type=str
    )
    parser.add_argument(
        '--model_name',
        type=str
    )
    args = parser.parse_args()
    main(args)
