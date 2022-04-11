#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A character-level LSTM modeled after:
https://www.kaggle.com/code/francescapaulin/character-level-lstm-in-pytorch/notebook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

class RNN(nn.Module):

    def __init__(self, chars, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.lr = lr

        self.lstm = nn.LSTM(len(chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(chars))

    def forward(self, inp, hidden):
        output, hidden = self.lstm(inp, hidden)
        output = self.dropout(output)
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
        )

class Model(object):

    def __init__(self, training_data, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        fin = open(training_data)
        data = fin.read()
        self.chars = tuple(set(data))
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.encoded = np.array([self.char_to_idx[ch] for ch in data])
        del data
        fin.close()

        self.rnn = RNN(self.chars, n_hidden, n_layers, drop_prob, lr)

        print(
            f"Initialized model with {len(self.encoded):,} characters",
            f"({len(self.chars)} unique)",
        )

    @staticmethod
    def one_hot_encode(arr, n_labels):
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return one_hot

    @staticmethod
    def get_batches(arr, batch_size, seq_length):
        batch_size_total = batch_size * seq_length
        n_batches = len(arr) // batch_size_total

        arr = arr[:n_batches * batch_size_total]
        arr = arr.reshape((batch_size, -1))

        for n in range(0, arr.shape[1], seq_length):
            x = arr[:, n:n+seq_length]
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y

    def predict(self, char, h=None, top_k=None):
        x = np.array([[self.char_to_idx[char]]])
        x = self.one_hot_encode(x, len(self.chars))
        inputs = torch.from_numpy(x)

        h = tuple([each.data for each in h])
        out, h = self.rnn(inputs, h)

        p = F.softmax(out, dim=1).data

        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        return self.idx_to_char[char], h

    def sample(self, size, prime='A', top_k=None):
        self.rnn.eval()

        chars = [ch for ch in prime]
        h = self.rnn.init_hidden(1)
        for ch in prime:
            char, h = self.predict(ch, h, top_k=top_k)

        chars.append(char)

        for i in range(size):
            char, h = self.predict(chars[-1], h, top_k=top_k)
            chars.append(char)

        return ''.join(chars)

    def train(self, epochs=3, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
        self.rnn.train()

        opt = torch.optim.Adam(self.rnn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        val_idx = int(len(self.encoded) * (1 - val_frac))
        data, val_data = self.encoded[:val_idx], self.encoded[val_idx:]

        counter = 0
        train_losses = []
        train_perplexities = []
        n_chars = len(self.chars)
        print("Training...")
        for e in range(epochs):
            h = self.rnn.init_hidden(batch_size)
            for x, y in self.get_batches(data, batch_size, seq_length):
                counter += 1
                x = self.one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                h = tuple([each.data for each in h])
                self.rnn.zero_grad()

                output, h = self.rnn(inputs, h)
                loss = criterion(output, targets.view(batch_size*seq_length))
                loss.backward()

                train_losses.append(loss.item())
                perplexity = torch.exp(loss)
                train_perplexities.append(perplexity.item())

                nn.utils.clip_grad_norm_(self.rnn.parameters(), clip)
                opt.step()

                if counter % print_every == 0:
                    val_h = self.rnn.init_hidden(batch_size)
                    self.rnn.eval()

                    val_losses = []
                    val_perplexities = []
                    for xx, yy in self.get_batches(val_data, batch_size, seq_length):
                        xx = self.one_hot_encode(xx, n_chars)
                        inputs, targets = torch.from_numpy(xx), torch.from_numpy(yy)

                        val_h = tuple([each.data for each in val_h])

                        output, val_h = self.rnn(inputs, val_h)
                        val_loss = criterion(output, targets.view(batch_size*seq_length))

                        val_losses.append(val_loss.item())
                        val_perplexity = torch.exp(val_loss)
                        val_perplexities.append(val_perplexity.item())

                    print("----------------------------------------------")
                    prime = np.random.choice(self.chars, 1)
                    text = self.sample(20, prime=prime)
                    print(text)
                    print("----------------------------------------------")
                    print(
                        f"Epoch {e+1}/{epochs}, Step: {counter}\n"
                        f"+ Loss: {loss.item():0.4f}..."
                        f"Val. loss: {np.mean(val_losses):0.4f}\n"
                        f"+ Perplexity: {perplexity.item():0.4f}..."
                        f"Val. perplexity: {np.mean(val_perplexities):0.4f}"
                    )

                    self.rnn.train()
                    
        self.training_info = {
            'iter': list(range(len(train_losses))),
            'loss': train_losses,
            'perp': train_perplexities
        }

    def save_training_info(self, filepath):
        with open(filepath, 'w') as j:
            json.dump(self.training_info, j)

