#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from charmodel.model import CharRNN

rnn = CharRNN('sherlock.txt', 100, 25, 1e-1)
rnn.train()

for m in ['loss', 'perp']:
    rnn.plot_performance(metric)
