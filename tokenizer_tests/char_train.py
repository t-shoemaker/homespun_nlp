#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from charmodel.model import CharRNN
from charmodel.utils import plot_performance
from charmodel.utils import load_rnn, save_rnn

def main(args):
    rnn = CharRNN(args.training_data, 100, 25, 1e-1)
    rnn.train(n_iters=args.n_iters)

    for metric in ['loss', 'perp']:
        plot_performance(rnn, model_name=args.model_name, metric=metric)

    rnn.save_training_info(args.model_name + '_info.json')
    save_rnn(rnn, args.model_name)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--training_data',
        type=str
    )
    parser.add_argument(
        '--n_iters',
        type=int
    )

    parser.add_argument(
        '--model_name',
        type=str
    )
    args = parser.parse_args()
    main(args)
