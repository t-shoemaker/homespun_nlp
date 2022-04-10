#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json

class CharRNN(object):   
    """This is a minimal character-level Vanilla RNN model, written by Andrej Karpathy.
    
    Original: https://gist.github.com/karpathy/d4dee566867f8291f086
    """
    
    def __init__(self, filepath, hidden_size, seq_length, learning_rate):
        fin = open(filepath, 'r')
        self.data = fin.read()
        self.chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        fin.close()
        
        # size of the hidden layer
        self.hidden_size = hidden_size
        # number of steps to unroll the RNN for
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # input to hidden
        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        # hidden to hidden
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        # hidden to output
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        # hidden bias
        self.bh = np.zeros((self.hidden_size, 1))
        # output bias
        self.by = np.zeros((self.vocab_size, 1))

        # static loss score
        self.loss = 0
        # static perplexity score
        self.perplexity = 0
        # a static hidden state for character generation
        self.hprev = np.zeros((self.hidden_size, 1))

    def calculate_loss(self, inputs, targets, hprev):
        """Calculate the loss for a pass.
        
        targets and inputs are lists of integers (chars from the training data)
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients of the model parameters, and the last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        
        # forward pass
        for t in range(len(inputs)):
            # encode in 1-of-k representation
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][inputs[t]] = 1
            # hidden state
            hs[t] = np.tanh(
                np.dot(self.Wxh, xs[t]) + 
                np.dot(self.Whh, hs[t - 1]) + 
                self.bh
            )
            # unnormalized log probabilities for next chars
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            # probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            # softmax (cross-entropy loss)
            loss += -np.log(ps[t][targets[t], 0])
        
        # backwards pass
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            # backprop into y
            dy[targets[t]] -= 1
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            # backprop into h
            dh = np.dot(self.Why.T, dy) + dhnext
            # backprop through tanh nonlinearity
            dhraw = (1 - hs[t] * hs[t]) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t - 1].T)
            dhnext = np.dot(self.Whh.T, dhraw)
        
        # clip to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
            
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]
    
    def sample(self, h, seed_idx, n_char):
        """Sample a sequence of integers from the model.
        
        h is memory state, seed_idx is the seed letter for the first time step
        n_char is the number of characters to sample
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        idxes = []
        for t in range(n_char):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            idxes.append(idx)
            
        return idxes
    
    def train(self, n_iters=10000, sample_rate=100, sample_size=200):
        """Train the model."""
        # set up a performance metric tracker
        self.training_info = {'iter': [], 'loss': [], 'perp': []}

        # n is iters, p is data pointer
        n, p = 0, 0
        mWxh = np.zeros_like(self.Wxh)
        mWhh = np.zeros_like(self.Whh)
        mWhy = np.zeros_like(self.Why)
        # memory variables for Adagrad
        mbh = np.zeros_like(self.bh)
        mby = np.zeros_like(self.by)
        # loss at iteration 0
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        
        while True:
            # prepare inputs (sweeps from left->right in steps seq_length long)
            if p + self.seq_length + 1 >= self.data_size or n == 0:
                # reset RNN memory
                hprev = np.zeros((self.hidden_size, 1))
                # go to start of the data
                p = 0
                
            inputs = [self.char_to_idx[ch] for ch in self.data[p:p+self.seq_length]]
            targets = [self.char_to_idx[ch] for ch in self.data[p+1:p+self.seq_length+1]]
            
            # sample for logging purposes
            if n % sample_rate == 0:
                sample_idx = self.sample(hprev, inputs[0], sample_size)
                txt = ''.join(self.idx_to_char[idx] for idx in sample_idx)
                print(f"=======\n{txt}\n=======\n")
            
            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.calculate_loss(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            # print progress and
            if n % sample_rate == 0:
                # update the static hidden state
                self.hprev = np.copy(hprev)
                # calculate perplexity, which can apparently be done from the cross-entropy, as per
                # https://stackoverflow.com/questions/61988776/how-to-calculate-perplexity-for-a-language-model-using-pytorch
                self.perplexity = np.exp(smooth_loss)
                self.loss = smooth_loss

                self.training_info['iter'].append(n)
                self.training_info['loss'].append(self.loss)
                self.training_info['perp'].append(self.perplexity)
                print(f"+ iter {n}\n+ loss: {self.loss:0.6f}\n+ perplexity: {self.perplexity:0.6f}\n")
            
            # perform parameter update with Adagrad
            for param, dparam, mem in zip(
                [self.Wxh, self.Whh, self.Why, self.bh, self.by],
                [dWxh, dWhh, dWhy, dbh, dby],
                [mWxh, mWhh, mWhy, mbh, mby]
            ):
                mem += dparam * dparam
                # adagrad update
                param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8)
            
            # move data pointer
            p += self.seq_length
            # iteration counter
            n += 1
            if n > n_iters:
                break

    def generate(self, char, n_chars=200):
        char_idx = self.char_to_idx[char]
        sample_idx = self.sample(self.hprev, char_idx, n_chars)
        txt = char + ''.join(self.idx_to_char[idx] for idx in sample_idx)
        return txt
            
    def save_training_info(self, filepath):
        with open(filepath, 'w') as j:
            json.dump(self.training_info, j)
