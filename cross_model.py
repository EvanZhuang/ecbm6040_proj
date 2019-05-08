from torch import nn
import torch
from cross_dataset import DataSet
from inspect import isclass


class RnnModel(nn.Module):
    def __init__(self, vocabulary, emb_dim, hidden_size):
        assert isinstance(vocabulary, DataSet.Vocabulary)
        super(RnnModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary.count,
                                      embedding_dim=emb_dim, padding_idx=vocabulary.pad_token).cuda()
        # self.rnn = nn.RNN(hidden_size=hidden_size, input_size=emb_dim, dropout=0.2, batch_first=True)
        self.rnn = nn.LSTM(hidden_size=hidden_size, input_size=emb_dim,
                           bidirectional=False, batch_first=True, num_layers=1).cuda()
        self.hidden = nn.Linear(in_features=hidden_size, out_features=8).cuda()
        self.output = nn.Linear(in_features=8, out_features=2).cuda()
        self.activation = nn.LogSoftmax(dim=-1).cuda()

    def forward(self, sequences, masks, *input):
        embs = self.embedding(sequences)
        embs = nn.utils.rnn.pack_padded_sequence(embs, masks, batch_first=True)
        output, (h_n, c_n) = self.rnn(embs)
        h_n = h_n.view(sequences.shape[0], -1)
        out = nn.functional.relu(self.hidden(h_n), inplace=False)
        out = self.output(out)
        out = self.activation(out)
        return h_n, out


class ConvModel(nn.Module):
    def __init__(self, vocab, emb_dim, hidden_size=32):
        super(ConvModel, self).__init__()
        assert isinstance(vocab, DataSet.Vocabulary)
        self.emb = nn.Embedding(num_embeddings=vocab.count,
                                      embedding_dim=emb_dim, padding_idx=vocab.pad_token).cuda()
        self.num_out_kernel = 16
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.num_out_kernel, kernel_size=(3,emb_dim)).cuda()
        #self.pool = nn.AvgPool2d(kernel_size = 1)
        self.hidden = nn.Linear(in_features=16, out_features=8).cuda()
        self.output = nn.Linear(in_features=8, out_features=2).cuda()
        self.activation = nn.LogSoftmax(dim=-1).cuda()

    def forward(self, sentences, masks, *input):
        emb = self.emb(sentences)
        emb = emb.unsqueeze(dim=1)
        cs = nn.functional.relu(self.conv(emb), inplace=False)
        cs = cs.view(sentences.shape[0], self.num_out_kernel, -1)
        mp = nn.AvgPool1d(kernel_size=cs.shape[-1])
        rs = mp(cs)
        rs = rs.view(sentences.shape[0], self.num_out_kernel)
        hs = nn.functional.relu(self.hidden(rs), inplace=False)
        hs = self.output(hs)
        soft = self.activation(hs)
        return rs, soft


class BiRnnModel(nn.Module):
    def __init__(self, vocabulary, emb_dim, hidden_size):
        assert isinstance(vocabulary, DataSet.Vocabulary)
        super(BiRnnModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary.count,
                                      embedding_dim=emb_dim, padding_idx=vocabulary.pad_token).cuda()
        # self.rnn = nn.RNN(hidden_size=hidden_size, input_size=emb_dim, dropout=0.2, batch_first=True)
        self.rnn = nn.LSTM(hidden_size=hidden_size, input_size=emb_dim,
                           bidirectional=True, batch_first=True, num_layers=1).cuda()
        self.hidden = nn.Linear(in_features=hidden_size*2, out_features=8).cuda()
        self.output = nn.Linear(in_features=8, out_features=2).cuda()
        self.activation = nn.LogSoftmax(dim=-1).cuda()

    def forward(self, sequences, masks, *input):
        embs = self.embedding(sequences)
        embs = nn.utils.rnn.pack_padded_sequence(embs, masks, batch_first=True)
        output, (h_n, c_n) = self.rnn(embs)
        #h_n = h_n.view(sequences.shape[0], -1)
        h_n = torch.cat([hh for hh in h_n], -1)
        out = nn.functional.relu(self.hidden(h_n), inplace=False)
        out = self.output(out)
        out = self.activation(out)
        return h_n, out
