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

class RcModel(nn.Module):
    """
    Relation Classification
    """
    def __init__(self, vocab_size, word_embedding_dim, entity_embedding_dim,
                 entity_num, word_hidden_dim, entity_hidden_dim, hidden_dim, 
                 batch_size, lstm_hidden_dim=0, use_attention=False, drop_out_rate=0.5):
        super(RcModel, self).__init__()

        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.entity_embedding_dim = entity_embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.entity_hidden_dim = entity_hidden_dim
        self.hidden_dim = hidden_dim
        self.entity_num = entity_num
        self.use_attention = use_attention
        self.batch_size = batch_size

        self.loss_fct = CrossEntropyLoss()

        self.dropout = nn.Dropout(drop_out_rate)

        if lstm_hidden_dim:
            self.bilstm = BiLSTM(vocab_size, word_embedding_dim, lstm_hidden_dim, batch_size, drop_out_rate)
            self.attn = Attention(dimensions=lstm_hidden_dim)
            self.fc_w = nn.Linear(lstm_hidden_dim, word_hidden_dim)
        else:
            self.word_embeds = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=1)
            self.attn = Attention(dimensions=word_embedding_dim)
            self.fc_w = nn.Linear(word_embedding_dim, word_hidden_dim)

        self.entity_embeds = nn.Embedding(entity_num, entity_embedding_dim)
        
        self.fc_e = nn.Linear(entity_embedding_dim, entity_hidden_dim)

        self.fc_c = nn.Linear(word_hidden_dim*3 + entity_hidden_dim, hidden_dim)
        
        self.classifier  = nn.Linear(hidden_dim, 2)

        self.tanh = nn.Tanh()
        # self.activation = nn.ReLU()
        self.activation = gelu

    def attn_sum(self, x):
        # x : [seq_length, 1, dimensions]
        x = torch.transpose(x, 0, 1)
        out, _ = self.attn(x, x)
        return out.sum(dim=1)

    def forward(self, inputs, masks, entities, labels):
        """
        inputs : (batch, seq_length)
        masks : (batch, seq_length)
        entities : (batch)
        labels : (batch)
        """

        if self.lstm_hidden_dim:
            inputs_feature = self.bilstm(inputs)
        else:
            inputs_feature = self.word_embeds(inputs)

        inputs_feature = self.dropout(inputs_feature)

        # get context representation
        if self.use_attention:
            obj_list = [self.attn_sum(out[(mask == 1).nonzero()])
                                    for out, mask in zip(inputs_feature, masks)]

            btwn_list = [self.attn_sum(out[(mask == 2).nonzero()])
                                    for out, mask in zip(inputs_feature, masks)]

            attr_list = [self.attn_sum(out[(mask == 3).nonzero()])
                                    for out, mask in zip(inputs_feature, masks)]

        else: # max pooling         
            obj_list = [out[(mask == 1).nonzero().squeeze()].view(-1, self.word_embedding_dim).max(0)[0].unsqueeze(0) 
                                    for out, mask in zip(inputs_feature, masks)]

            btwn_list = [out[(mask == 2).nonzero().squeeze()].view(-1, self.word_embedding_dim).max(0)[0].unsqueeze(0) 
                                    for out, mask in zip(inputs_feature, masks)]

            attr_list = [out[(mask == 3).nonzero().squeeze()].view(-1, self.word_embedding_dim).max(0)[0].unsqueeze(0) 
                                    for out, mask in zip(inputs_feature, masks)]

        obj = torch.cat(obj_list)
        obj = self.activation(self.fc_w(obj))
        obj = self.dropout(obj)
        
        btwn = torch.cat(btwn_list)
        btwn = self.activation(self.fc_w(btwn))
        btwn = self.dropout(btwn)

        attr = torch.cat(attr_list)
        attr = self.activation(self.fc_w(attr))
        attr = self.dropout(attr)

        ent = self.entity_embeds(entities)
        ent = self.activation(self.fc_e(ent))
        ent = self.dropout(ent)

        # concat
        features = torch.cat([obj, btwn, attr, ent], dim=-1)

        features = self.activation(self.fc_c(features))
        features = self.dropout(features)

        out = self.classifier(features)  # out : (batch, 2)

        loss = self.loss_fct(out, labels)

        return out, loss
