from cross_dataset import DataSet
import numpy as np
import math
import torch

def train(model, criterion, optim, data, epochs):
    #assert isinstance(data, DataSet)
    total = len(data.entries)
    for epoch in range(epochs):
        sequences, masks, labels = data.get_next_batch_train_data()
        hidden, output = model(sequences, masks)
        optim.zero_grad()
        loss = criterion(output, labels)
        #print(str(epoch) + ' , ' + str(loss.item()))
        loss.backward()
        optim.step()


def decode(model, data, batch_size = 32):
    # print(input_sentences)
    test_batches = len(data.test_batch_indices)
    print(test_batches)
    sequences, masks, labels = data.get_next_batch_test_data()
    hidden, output = model(sequences, masks)
    topv, topi = output.data.topk(1)
    output_r = topi.squeeze()
    output_t = labels
    for i in range(test_batches-1):
        sequences, masks, labels = data.get_next_batch_test_data()
        hidden, output = model(sequences, masks)
        topv, topi = output.data.topk(1)
        output_r = torch.cat((output_r, topi.squeeze()))
        output_t = torch.cat((output_t, labels))
    return output_r.cpu().data.numpy(), output_t.cpu().data.numpy()


def decode_train(model, data, batch_size = 32):
    # print(input_sentences)
    data.initialize_batch()
    test_batches = len(data.batch_indices)
    print(test_batches)
    sequences, masks, labels = data.get_next_batch_train_data()
    hidden, output = model(sequences, masks)
    topv, topi = output.data.topk(1)
    output_r = topi.squeeze()
    output_t = labels
    for i in range(test_batches-1):
        sequences, masks, labels = data.get_next_batch_train_data()
        hidden, output = model(sequences, masks)
        topv, topi = output.data.topk(1)
        output_r = torch.cat((output_r, topi.squeeze()))
        output_t = torch.cat((output_t, labels))
    return output_r.cpu().data.numpy(), output_t.cpu().data.numpy()


def decode_test(model, data, batch_size = 32):
    # print(input_sentences)
    test_batches = len(data.test_batch_indices)
    print(test_batches)
    tok_lst = []
    
    sequences, masks, labels = data.get_next_batch_test_data()
    hidden, output = model(sequences)
    topv, topi = output.data.topk(1)
    output_r = topi.squeeze()
    output_t = labels
    seq = sequences.cpu().data.numpy()        
    for _ in seq:
        tmp_lst = []
        for i in range(_.shape[0]):
            if data.vocab.get_token(_[i]) != '<PAD>':
                tmp_lst.append(data.vocab.get_token(_[i]))
        tok_lst.append(' '.join(tmp_lst))
    for i in range(test_batches-1):
        sequences, masks, labels = data.get_next_batch_test_data()
        hidden, output = model(sequences)
        topv, topi = output.data.topk(1)
        output_r = torch.cat((output_r, topi.squeeze()))
        output_t = torch.cat((output_t, labels))
        seq = sequences.cpu().data.numpy()        
        for _ in seq:
            tmp_lst = []
            for i in range(_.shape[0]):
                if data.vocab.get_token(_[i]) != '<PAD>':
                    tmp_lst.append(data.vocab.get_token(_[i]))
            tok_lst.append(' '.join(tmp_lst))
    return tok_lst, output_r.cpu().data.numpy(), output_t.cpu().data.numpy()


def accuracy_score(original, predicted):
    total = len(original)
    correct = 0.
    assert total == len(predicted), 'Number mismatch predicted vs original'
    for a, b in zip(original, predicted):
        if a == b:
            correct += 1.0
    return float(correct) / total * 100.0

def examine():

    return
