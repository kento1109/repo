import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data, datasets
from torchcrf import CRF
from torchhelper import FloatTensor, LongTensor, zeros, Attention
import optuna

CUDA = True if torch.cuda.is_available() else False

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x


# Create model
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,
                 batch_size, drop_out_rate=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim

        # self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        self.drop_out_rate = drop_out_rate

        # if trial is not None:
        #     self.hidden_dim = trial.suggest_discrete_uniform('hidden_dim', 128, 256, 128)
        #     # self.drop_out_rate = trial.suggest_categorical('drop_out_rate', drop_out_rate)
        #     # self.batch_size = trial.suggest_categorical('batch_size', batch_size)
        # else:
        self.hidden_dim = hidden_dim
        self.drop_out_rate = drop_out_rate
        self.batch_size = batch_size

        self.dropout = nn.Dropout(self.drop_out_rate)

        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim,
                                        padding_idx=1)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # self.attention = Attention(hidden_dim)

        # Maps the output of the LSTM into tag space.
        self.emissons = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()

        self.crf = CRF(len(self.tag_to_ix))

    def init_hidden(self):
        return (zeros(2, self.batch_size, self.hidden_dim // 2),
                zeros(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        """
        sentence : (sent, batch)
        """
        # Initialise hidden state
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        if self.drop_out_rate:
            embeds = self.dropout(embeds)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # lstm_out, _ = self.attention(lstm_out, lstm_out)

        if self.drop_out_rate:
            lstm_out = self.dropout(lstm_out)
        lstm_feats = self.emissons(lstm_out)
        return lstm_feats

    def forward(self, sentence, tags):
        """
        sentence : (sent, batch)
        tags : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        # Computing log likelihood
        mask = sentence.ne(1)  # (s, b)
        llh = self.crf(emissions, tags, mask=mask)
        return - llh

    def predict(self, sentence):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask)

    def predict_prob(self, sentence, n_best=1):
        """
        sentence : (sent, batch)
        """
        self.batch_size = sentence.size(1)
        mask = sentence.ne(1)  # (s, b)
        # Get the emission scores from the BiLSTM
        emissions = self._get_lstm_features(sentence)
        return self.crf.decode(emissions, mask=mask, prob=True, n_best=n_best)

