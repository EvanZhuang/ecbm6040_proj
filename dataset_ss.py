import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

class DataEntry:
    def __init__(self, dataset, sentence, label, meta_data = None, parser = None):
        self.dataset = dataset
        assert isinstance(self.dataset, DataSet)
        self.sentence = sentence
        self.label = label
        if parser is not None:
            self.words = parser.parse(self.sentence)
        else:
            self.words = self.sentence.split()
        self.meta_data = meta_data
        pass

    def init_word_index(self):
        assert self.dataset.vocab is not None, 'Initialize Dataset Vocabulary First'
        self.word_indices = [self.dataset.vocab.start_token]
        for word in self.words:
            self.word_indices.append(self.dataset.vocab.get_token_id(word))
        self.word_indices.append(self.dataset.vocab.end_token)

    def __repr__(self):
        return str(self.word_indices) + '\t' + str(self.label)


class DataSet:
    class Vocabulary:
        def __init__(self):
            self.start_token = 0
            self.end_token = 1
            self.pad_token = 2
            self.unk = 3
            self.index_to_token = {0: "<START>", 1:"<END>", 2:"<PAD>", 3: "<UNK>"}
            self.token_to_index = {"<START>":0, "<END>":1, "<PAD>":2, "<UNK>":3}
            self.count = 4
            pass

        def get_token_id(self, token):
            if token in self.token_to_index.keys():
                return self.token_to_index[token]
            else:
                return self.unk

        def get_token(self, id):
            assert id < self.count, 'Invalid token ID'
            return self.index_to_token[id]

        def add_token(self, token):
            index = self.get_token_id(token)
            if index != self.unk:
                return index
            else:
                index = self.count
                self.count += 1
                self.index_to_token[index] = token
                self.token_to_index[token] = index
                return index

    def __init__(self):
        self.entries = []
        pass

    def add_data_entry(self, entry):
        assert isinstance(entry, DataEntry)
        self.entries.append(entry)

    def init_data_set(self, vocab_size=1000, batch_size=32, test_percentage=0.2):
        self.build_vocabulary(vocab_size)
        for entry in self.entries:
            entry.init_word_index()
        self.batch_size = batch_size
        self.split_train_test(test_percentage)
        self.initialize_batch()
        self.initialize_unsup_batch()
        self.initialize_test_batch()

    def split_train_test(self, test_p, seed=42):
        import pandas as pd
        np.random.seed(seed=seed)
        self.max_seq_len = max([len(self.entries[id].word_indices) for id in np.arange(0,len(self.entries)-1, 1)])
        self.test_entries = []
        self.unsupervised_entries = []
        print("Raw Data Size:", len(self.entries))
        entry_cp = self.entries[:]
        for entry in entry_cp:
            prob = np.random.uniform()
            if entry.meta_data == -1:
                self.unsupervised_entries.append(entry)
                self.entries.remove(entry)
            elif prob < test_p:
                self.test_entries.append(entry)
                self.entries.remove(entry)
        self.sup_num = len(self.entries)
        self.train_data_size = len(self.entries) + len(self.unsupervised_entries)


    def build_vocabulary(self, vocab_size=2000):
        self.vocab = DataSet.Vocabulary()
        words = {}
        total_words = 0
        for entry in self.entries:
            for word in entry.words:
                if word in words.keys():
                    words[word] += 1
                else:
                    words[word] = 1
                total_words += 1
        word_freq = [[key, words[key]] for key in words.keys()]
        word_freq = sorted(word_freq, key=lambda x:x[1], reverse=True)
        accepted_words = word_freq[:vocab_size]
        for word, count  in accepted_words:
            self.vocab.add_token(word)
        print('Total Number of Words', total_words)
        print('Unique Words : ', len(words.keys()))
        print('Vocab Size : ', len(accepted_words))
    

    def get_dataset_by_ids(self, ids):
        max_seq_len = max([len(self.entries[id].word_indices) for id in ids])
        ids.sort(key = lambda x: len(self.entries[x].word_indices),reverse=True)
        token_indices = []
        masks = []
        labels = []
        for index in ids:
            indices = [self.vocab.pad_token] * max_seq_len
            mask = [0] * max_seq_len
            for i, w_index in enumerate(self.entries[index].word_indices):
                indices[i] = w_index
                mask[i] = 1
            token_indices.append(indices)
            masks.append(mask)
            labels.append(self.entries[index].label)
        return Variable(torch.LongTensor(np.asarray(token_indices)).cuda()), \
               Variable(torch.LongTensor(np.asarray(masks)).cuda()), \
               Variable(torch.LongTensor(np.asarray(labels)).cuda()), max_seq_len

    def get_unsup_dataset_by_ids(self, ids):
        max_seq_len = max([len(self.unsupervised_entries[id].word_indices) for id in ids])
        ids.sort(key = lambda x: len(self.unsupervised_entries[x].word_indices),reverse=True)
        token_indices = []
        masks = []
        labels = []
        for index in ids:
            indices = [self.vocab.pad_token] * max_seq_len
            mask = [0] * max_seq_len
            for i, w_index in enumerate(self.unsupervised_entries[index].word_indices):
                indices[i] = w_index
                mask[i] = 1
            token_indices.append(indices)
            masks.append(mask)
            labels.append(self.unsupervised_entries[index].label)
        return Variable(torch.LongTensor(np.asarray(token_indices)).cuda()), \
               Variable(torch.LongTensor(np.asarray(masks)).cuda()), \
               Variable(torch.LongTensor(np.asarray(labels)).cuda()), max_seq_len

    def get_test_dataset_by_ids(self, ids):
        max_seq_len = max([len(self.test_entries[id].word_indices) for id in ids])
        ids.sort(key = lambda x: len(self.test_entries[x].word_indices),reverse=True)
        token_indices = []
        masks = []
        labels = []
        for index in ids:
            indices = [self.vocab.pad_token] * max_seq_len
            mask = [0] * max_seq_len
            for i, w_index in enumerate(self.test_entries[index].word_indices):
                indices[i] = w_index
                mask[i] = 1
            token_indices.append(indices)
            masks.append(mask)
            labels.append(self.test_entries[index].label)
        return Variable(torch.LongTensor(np.asarray(token_indices)).cuda()), \
               Variable(torch.LongTensor(np.asarray(masks)).cuda()), \
               Variable(torch.Tensor(np.asarray(labels)).cuda())

    def initialize_batch(self):
        total = len(self.entries)
        indices = list(np.arange(0,total-1, 1))
        np.random.shuffle(indices)
        self.batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + self.batch_size
            if c_end > end:
                c_end = end
            self.batch_indices.append(indices[curr:c_end])
            curr = c_end
        # print ('Number of batches : ', len(self.batch_indices))

    def initialize_unsup_batch(self):
        total = len(self.unsupervised_entries)
        indices = list(np.arange(0,total-1, 1))
        np.random.shuffle(indices)
        self.unsup_batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + self.batch_size
            if c_end > end:
                c_end = end
            self.unsup_batch_indices.append(indices[curr:c_end])
            curr = c_end

    def initialize_test_batch(self):
        total = len(self.test_entries)
        indices = list(np.arange(0,total-1, 1))
        np.random.shuffle(indices)
        self.test_batch_indices = []
        start = 0
        end = len(indices)
        curr = start
        while curr < end:
            c_end = curr + self.batch_size
            if c_end > end:
                c_end = end
            self.test_batch_indices.append(indices[curr:c_end])
            curr = c_end
        # print ('Number of batches : ', len(self.batch_indices))

    def get_next_batch_train_data(self):
        if len(self.batch_indices) == 0:
            self.initialize_batch()
        indices = self.batch_indices[0]
        self.batch_indices = self.batch_indices[1:]
        return self.get_dataset_by_ids(indices)

    def get_next_batch_train_unsup_data(self):
        if len(self.unsup_batch_indices) == 0:
            self.initialize_unsup_batch()
        indices = self.unsup_batch_indices[0]
        self.unsup_batch_indices = self.unsup_batch_indices[1:]
        return self.get_unsup_dataset_by_ids(indices)

    def get_next_batch_test_data(self):
        if len(self.batch_indices) == 0:
            self.initialize_test_batch()
        indices = self.test_batch_indices[0]
        self.test_batch_indices = self.test_batch_indices[1:]
        return self.get_test_dataset_by_ids(indices)

    def get_test_data(self):
        max_seq_len = max([len(entry.word_indices) for entry in self.test_entries])
        token_indices = []
        masks = []
        labels = []
        for entry in self.test_entries:
            indices = [self.vocab.pad_token] * max_seq_len
            mask = [0] * max_seq_len
            for i, w_index in enumerate(entry.word_indices):
                indices[i] = w_index
                mask[i] = 1
            token_indices.append(indices)
            masks.append(mask)
            labels.append(entry.label)
        return Variable(torch.LongTensor(np.asarray(token_indices)).cuda()), \
               Variable(torch.LongTensor(np.asarray(masks)).cuda()), \
               Variable(torch.LongTensor(np.asarray(labels)).cuda())

    def get_complete_train_data(self):
        max_seq_len = max([len(entry.word_indices) for entry in self.entries])
        token_indices = []
        masks = []
        labels = []
        for entry in self.entries:
            indices = [self.vocab.pad_token] * max_seq_len
            mask = [0] * max_seq_len
            for i, w_index in enumerate(entry.word_indices):
                indices[i] = w_index
                mask[i] = 1
            token_indices.append(indices)
            masks.append(mask)
            labels.append(entry.label)
        return Variable(torch.LongTensor(np.asarray(token_indices)).cuda()), \
               Variable(torch.LongTensor(np.asarray(masks)).cuda()), \
               Variable(torch.LongTensor(np.asarray(labels)).cuda())