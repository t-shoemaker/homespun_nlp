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
        """A forward pass through the network."""
        # get outputs and the new hidden state from the lstm
        output, hidden = self.lstm(inp, hidden)
        # pass outputs through the dropout layer 
        output = self.dropout(output)
        # stack the outputs and put them through the fully-connected layer
        output = output.contiguous().view(-1, self.n_hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize a hidden state."""
        # two new tensors with size n_layers x batch_size * h_hidden---these are the hidden state
        # and the cell state of the lstm
        weight = next(self.parameters()).data
        return (
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
            weight.new(self.n_layers, batch_size, self.n_hidden).zero_()
        )

class Model(object):

    def __init__(
        self,
        training_data,
        shuffle=True,
        n_hidden=256,
        n_layers=2,
        drop_prob=0.5,
        lr=0.001
        ):
        # data should be a plaintext file, preferably multi-line
        fin = open(training_data)
        data = fin.read()
        # shuffle the lies of the file to break dependencies
        if shuffle:
            data = self._shuffle(data)

        # get all the unique characters, then build mappings from characters to their index
        # positions and back again
        self.chars = tuple(set(data))
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.encoded = np.array([self.char_to_idx[ch] for ch in data])
        del data
        fin.close()

        # initialize a network
        self.rnn = RNN(self.chars, n_hidden, n_layers, drop_prob, lr)

        # create an empty training stats dictionary
        self.training_info = None

        print(
            f"Initialized model with {len(self.encoded):,} characters",
            f"({len(self.chars)} unique)",
        )

    def _shuffle(self, data):
        """Shuffle lines."""
        data = data.split('\n')
        np.random.shuffle(data)
        return '\n'.join(data)

    @staticmethod
    def one_hot_encode(arr, n_labels):
        """Preprocess the data."""
        # initialize the array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
        # fill the array with one shot vals
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
        # reshape to return to the original
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return one_hot

    @staticmethod
    def get_batches(arr, batch_size, seq_length):
        """Create a batch generator."""
        # find the total number of batches that can be made
        batch_size_total = batch_size * seq_length
        n_batches = len(arr) // batch_size_total
        # keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size_total]
        arr = arr.reshape((batch_size, -1))

        for n in range(0, arr.shape[1], seq_length):
            # features
            x = arr[:, n:n+seq_length]
            # targets
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y

    def predict(self, char, h=None, top_k=None):
        """Predict n characters after an input string."""
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
        """Sample from the model using a primer character."""
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
        """Train the model."""
        # put the model into its training state
        self.rnn.train()
        # set up the optimizer and cross-entropy metric
        opt = torch.optim.Adam(self.rnn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        # create training and validation data
        val_idx = int(len(self.encoded) * (1 - val_frac))
        data, val_data = self.encoded[:val_idx], self.encoded[val_idx:]

        # number of steps
        counter = 0
        # arrays for logging the training stats
        train_losses = []
        train_perplexities = []

        n_chars = len(self.chars)
        print("Training...")
        for e in range(epochs):
            # intialize a hidden state
            h = self.rnn.init_hidden(batch_size)
            # make the batches and roll through them; seq_length is the number of characters to put
            # in a batch
            for x, y in self.get_batches(data, batch_size, seq_length):
                counter += 1

                # encode an input batch; convert it and the target to tensors
                x = self.one_hot_encode(x, n_chars)
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                # create new variables for the hidden state (otherwise the model would backpropagate
                # through the entire training history)
                h = tuple([each.data for each in h])
                # zero out the accumulated gradients
                self.rnn.zero_grad()

                # get the model output
                output, h = self.rnn(inputs, h)
                # calculate loss and do the backprop
                loss = criterion(output, targets.view(batch_size*seq_length))
                loss.backward()

                # calculate perplexity and log it along with the loss
                train_losses.append(loss.item())
                perplexity = torch.exp(loss)
                train_perplexities.append(perplexity.item())

                # clipping helps prevent exploding gradients
                nn.utils.clip_grad_norm_(self.rnn.parameters(), clip)
                opt.step()

                # print a training log
                if counter % print_every == 0:
                    # the process is the same as above, though the model must be put into its
                    # evaluation state (don't forget to put it back into training!)
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
                    # select a random character from the data and use it to generate a sample
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

                    # put the model back into its training state
                    self.rnn.train()

        # load the training stats into a dictionary
        self.training_info = {
            'iter': list(range(len(train_losses))),
            'loss': train_losses,
            'perp': train_perplexities
        }

    def save_training_info(self, filepath):
        """Save the training stats."""
        with open(filepath, 'w') as j:
            json.dump(self.training_info, j)

