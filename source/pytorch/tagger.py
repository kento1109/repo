import os
import sys
sys.path.append('/data/sugimoto/xray/src')
import subprocess
import configparser
import xray_utils
from xray_utils import calc_acc, remove_true_labels, sents_itos, softmax, read_conll2txt
from xray_utils import get_true_labels, get_sents, get_examples, get_label_name
from xray_utils import sequence_to_csv, most_similar, check_wrong_result, cos_sim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import csv
import copy
import pandas as pd
from torchtext import data, datasets
import torchdataset
from tensorboardX import SummaryWriter
from model import BiLSTM_CRF, BiLSTM_CRF_Char, BiLSTM_CNN_CRF, BiLSTM_CRF_Flair
from torchhelper import argmax, log_sum_exp, prepare_sequence
from seqeval.metrics import classification_report, f1_score
import numpy as np
import random
import logging
import shutil
import pickle

config = configparser.ConfigParser()
config.read('/data/sugimoto/experiments/experiment.ini')

CT_DIR = config.get('FILE', 'ct_dir')
TRAIN_FILE = config.get('FILE', 'train_file')
VALID_FILE = config.get('FILE', 'valid_file')
TEST_FILE = config.get('FILE', 'test_file')
PREDICT_FILE = config.get('FILE', 'predict_file')
RESULT_FILE = config.get('FILE', 'result_file')
OUT_FILE = config.get('FILE', 'out_file')
MODEL_DIR = config.get('FILE', 'model_dir')
MODEL_FILE = config.get('FILE', 'model_file')
LABEL_FILE = config.get('FILE', 'label_file')
VOCAB_FILE = config.get('FILE', 'vocab_file')
MODEL_NAME = config.get('FILE', 'model_name')
MYSEED = config.getint('EXPERIMENT', 'my_seed')
BATCH_SIZE =  config.getint('EXPERIMENT', 'batch_size')
DROP_OUT_RATE = config.getfloat('EXPERIMENT', 'drop_out_rate')
CHAR_EMBEDDING_DIM = config.getint('EXPERIMENT', 'char_embedding_dim')
HIDDEN_DIM =  config.getint('EXPERIMENT', 'hidden_dim')
EMBEDDING_DIM =  config.getint('EXPERIMENT', 'embedding_dim')
USE_TRAIN_RATIO = config.getfloat('EXPERIMENT', 'use_train_ratio')
EPOCHS =  config.getint('EXPERIMENT', 'epochs')
LL =  config.getfloat('EXPERIMENT', 'learning_late')
PRETRAINED_VECTOR = config.getboolean('EXPERIMENT', 'pretrained_vector')

CUDA = True if torch.cuda.is_available() else False

if os.path.isdir("runs/" + MODEL_NAME): shutil.rmtree("runs/" + MODEL_NAME)
writer = SummaryWriter("runs/" + MODEL_NAME)

logger = logging.getLogger('tagger')

def set_seed(my_seed):
  torch.manual_seed(my_seed)
  torch.cuda.manual_seed(my_seed)
  np.random.seed(my_seed)
  random.seed(my_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def logging_init(verbose):
    global logger
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    logger.addHandler(ch)

    fh = logging.FileHandler('logs/result.log')
    ch.setLevel(log_level)
    logger.addHandler(fh)

    return logger

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def _train(model, sents, labels, optimizer, chars=None, flair_embs=None):
    model.train()
    optimizer.zero_grad()
    if chars is not None:
        loss = model(sents, chars, labels)
    elif flair_embs is not None:
        loss = model(sents, labels, flair_embs)
    else:
        loss = model(sents, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.item() / sents.size(1)

def _valid(model, sents, labels, chars=None, flair_embs=None):
    model.eval()
    with torch.no_grad():
        if chars is not None:
            loss = model(sents, chars, labels)
            pred = model.predict(sents, chars)
        elif flair_embs is not None:
            loss = model(sents, labels, flair_embs)
            pred = model.predict(sents, flair_embs)
        else:
            loss = model(sents, labels)
            pred = model.predict(sents)
        acc = calc_acc(torch.t(labels).cpu().numpy(), pred)
    return loss.item() / sents.size(1), acc

def _predict(model, test_iter, use_flair):
    model.eval()
    with torch.no_grad():
        if test_iter.batch_size == 1:
            pred_list = []
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    sents = test_data[0].word
                    flair_embs = test_data[1]
                    pred_list.append(model.predict(sents, flair_embs)[0])
                elif hasattr(test_data, 'char'):
                    sents = test_data.word
                    chars = test_data.char
                    pred_list.append(model.predict(sents, chars)[0])
                else:
                    sents = test_data.word
                    pred_list.append(model.predict(sents)[0])
            return pred_list
        else:
            for i, test_data in enumerate(test_iter):
                if use_flair:
                    sents = test_data[0].word
                    flair_embs = test_data[1]
                    pred = model.predict(sents, flair_embs)
                elif hasattr(test_data, 'char'):
                    chars = test_data.char
                    pred = model.predict(sents, chars)
                else:
                    sents = test_data.word
                    pred = model.predict(sents, flair_embs)
            return pred

def evaluate(model,
             test_ds,
             word_vocab,
             label_vocab,
             each_f1_score=False,
             is_final=False,
             check_result_tag='',
             error_analysis=False,
             check_wrong_tag='',
             use_flair=False):

    batch_size = 1
    if use_flair:
        test_iter = torchdataset.BucketIteratorWithFlair(test_ds,
                                                         batch_size=batch_size,
                                                         sort_key=lambda x: len(x.word),
                                                         shuffle=False,
                                                         repeat=False,
                                                         device="cuda")
    else:
        test_iter = data.BucketIterator(test_ds,
                                        batch_size=batch_size,
                                        sort_key=lambda x: len(x.word),
                                        shuffle=False,
                                        repeat=False,
                                        device="cuda")

    docs_idx = get_sents(test_iter, use_flair)
    true_ldxes = get_true_labels(test_iter, use_flair=use_flair)
    true_labels = sents_itos(true_ldxes, label_vocab, omission=False)
    pred_idxes = _predict(model, test_iter, use_flair)
    pred_labels = sents_itos(pred_idxes, label_vocab, omission=False)

    if is_final:
        logger.debug(classification_report(true_labels, pred_labels, digits=4))
        # print(classification_report(true_labels, pred_labels, digits=4))
    # else:
    #     print('f1 score: {:.3f}'.format(f1_score(true_labels, pred_labels)))
    # if check_wrong_tag:
    #     docs = sents_itos(docs_idx, word_vocab)
    #     examples = get_examples(test_iter)
    #     print(check_wrong_result(docs, examples, true_labels, pred_labels,
    #                        tag=check_result_tag, topn=10, verbose=True, target=True))
    #     print(check_wrong_result(docs, examples, true_labels, pred_labels,
    #                        tag=check_result_tag, topn=10, verbose=True, target=False))
    if error_analysis:
        f = open(RESULT_FILE, 'w')
        writer = csv.writer(f, lineterminator='\n', delimiter=' ')
        docs = sents_itos(docs_idx, word_vocab)
        for sent, trues, preds in zip(docs, true_labels, pred_labels):
            if check_wrong_tag:
                if trues != preds:
                    if check_wrong_tag == 'all':
                        print('doc : ' + ' '.join(doc))
                        print('true: ' + ' '.join(true))
                        print('pred: ' + ' '.join(pred))
                        print('-'*100)
                    else:
                        check_wrong_result(doc, true, pred, check_wrong_tag)
            for token, true_label, pred_label in zip(sent, trues, preds):
                writer.writerow([token, true_label, pred_label])
            writer.writerow([])
        f.close()

    if each_f1_score:
        label_name = get_label_name(label_vocab, omission=True)
        score_dic = {}
        for l in label_name:
            score_dic[l] = f1_score(true_labels, pred_labels, target_name=l)
        return score_dic
    else:
        score = f1_score(true_labels, pred_labels)
        return score

def train(model_name,
          my_seed=1,
          verbose=False,
          train_file='',
          valid_file='',
          test_file='',
          save_model=False,
          each_f1_score=False,
          sep=' ',
          use_flair=False,
          embedding_dim=None,
          char_embedding_dim=None,
          hidden_dim=None,
          drop_out_rate=None,
          batch_size=None,
          trial=None
          ):

    # set target file
    global TRAIN_FILE
    global VALID_FILE
    global TEST_FILE
    if train_file:
        TRAIN_FILE = train_file
    if valid_file:
        VALID_FILE = valid_file
    if test_file:
        TEST_FILE = test_file

    global DROP_OUT_RATE
    if drop_out_rate is not None:
        DROP_OUT_RATE = trial.suggest_categorical('drop_out_rate', drop_out_rate)

    global HIDDEN_DIM
    if hidden_dim is not None:
        HIDDEN_DIM = trial.suggest_categorical('hidden_dim', hidden_dim)

    global BATCH_SIZE
    if batch_size is not None:
        BATCH_SIZE = trial.suggest_categorical('batch_size', batch_size)

    global EMBEDDING_DIM
    global VOCAB_FILE
    if embedding_dim is not None:
        EMBEDDING_DIM = trial.suggest_categorical('embedding_dim', embedding_dim)
        VOCAB_FILE = 'vocab/ct_word2vec_cbow_{}.vocab'.format(EMBEDDING_DIM)

    global CHAR_EMBEDDING_DIM
    if char_embedding_dim is not None:
        CHAR_EMBEDDING_DIM = trial.suggest_categorical('char_embedding_dim', char_embedding_dim)

    # set seed
    set_seed(my_seed)

    # logging initialize
    global logger
    logger = logging_init(verbose)

    # load vocabulary
    word_vocab = torch.load(VOCAB_FILE)

    # make dataset
    WORD = data.Field()
    # set vocab obj to WORD.vocab
    setattr(WORD, "vocab", word_vocab)

    LABEL = data.Field()

    if model_name in ('BiLSTM_CNN_CRF', 'BiLSTM_CRF_Char'):
        # for char field
        nesting_field = data.Field(tokenize=list)
        CHAR = data.NestedField(nesting_field)

        train_ds, valid_ds, test_ds = torchdataset.SequenceTaggingDataset_with_CHAR.splits(
            fields=[('word', WORD), ('char', CHAR), ('label', LABEL)],
            path='',
            train=TRAIN_FILE,
            validation=VALID_FILE,
            test=TEST_FILE,
            separator=sep
        )

        # build char vocab
        CHAR.build_vocab(train_ds, valid_ds, test_ds)
        char_vocab = CHAR.vocab

    else:
        train_ds, valid_ds, test_ds = datasets.SequenceTaggingDataset.splits(
            fields=[('word', WORD), ('label', LABEL)],
            path='',
            train=TRAIN_FILE,
            validation=VALID_FILE,
            test=TEST_FILE,
            separator=sep
        )

    train_size = int(len(train_ds) * USE_TRAIN_RATIO)
    train_ds.examples = train_ds.examples[:train_size]

    logger.debug('pre-trained    : {}'.format(PRETRAINED_VECTOR))
    logger.debug('num train data : {}'.format(len(train_ds)))
    logger.debug('num valid data : {}'.format(len(valid_ds)))
    logger.debug('num test data  : {}'.format(len(test_ds)))

    # build label vocab
    LABEL.build_vocab(train_ds, valid_ds, test_ds)
    label_vocab = LABEL.vocab
    torch.save(label_vocab, LABEL_FILE)

    if model_name == 'BiLSTM_CRF_Flair':

        # make batch
        train_iter, valid_iter, test_iter = torchdataset.BucketIteratorWithFlair.splits(
            (train_ds, valid_ds, test_ds),
            # batch_sizes=(BATCH_SIZE, len(valid_ds), len(test_ds)),
            batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
            sort_key=lambda x: len(x.word),
            shuffle=True,
            repeat=False,
            device="cuda")

        use_flair = True

    else:

        train_iter, valid_iter, test_iter = data.BucketIterator.splits(
            (train_ds, valid_ds, test_ds),
            batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
            sort_key=lambda x: len(x.word),
            shuffle=True,
            repeat=False,
            device="cuda")

    # EMBEDDING_DIM = word_vocab.vectors.size(1)

    if model_name == 'BiLSTM_CNN_CRF':
        model = BiLSTM_CNN_CRF(len(word_vocab), len(char_vocab),
                               label_vocab.stoi,
                               EMBEDDING_DIM, CHAR_EMBEDDING_DIM,
                               HIDDEN_DIM, BATCH_SIZE,
                               drop_out_rate=DROP_OUT_RATE)

    elif model_name == 'BiLSTM_CRF_Char':
        model = BiLSTM_CRF_Char(len(word_vocab), len(char_vocab),
                                label_vocab.stoi,
                                EMBEDDING_DIM, CHAR_EMBEDDING_DIM,
                                HIDDEN_DIM, HIDDEN_DIM, BATCH_SIZE,
                                drop_out_rate=DROP_OUT_RATE)

    elif model_name == 'BiLSTM_CRF_Flair':
        model = BiLSTM_CRF_Flair(len(word_vocab),
                                 label_vocab.stoi,
                                 EMBEDDING_DIM,
                                 HIDDEN_DIM, BATCH_SIZE,
                                 drop_out=True)

    else:
        model = BiLSTM_CRF(len(word_vocab),
                           label_vocab.stoi,
                           EMBEDDING_DIM,
                           HIDDEN_DIM, BATCH_SIZE,
                           drop_out_rate=DROP_OUT_RATE)


    if PRETRAINED_VECTOR:
        # set pre-trained vector
        model.word_embeds.weight.data.copy_(word_vocab.vectors)

    if CUDA:
        model = model.cuda()

    # optimizer = optim.RMSprop(model.parameters(), lr=LL)
    optimizer = optim.SGD(model.parameters(), lr=LL)

    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='min')

    best_val_loss = 999
    best_f1_score = 0
    # best_val_acc = 0
    curr_lr = LL
    total_train_loss = []
    total_valid_loss = []
    total_valid_acc = []
    best_model = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        train_loss = []
        valid_loss = []
        valid_acc = []

        for i, train_data in enumerate(train_iter):

            if model_name in ('BiLSTM_CNN_CRF', 'BiLSTM_CRF_Char'):
                sents = train_data.word
                labels = train_data.label
                chars = train_data.char
                _tr_loss = _train(model, sents, labels, optimizer, chars=chars)
            elif model_name == 'BiLSTM_CRF_Flair':
                sents = train_data[0].word
                labels = train_data[0].label
                flair_embs = train_data[1]
                _tr_loss = _train(model, sents, labels, optimizer, flair_embs=flair_embs)
            else:
                sents = train_data.word
                labels = train_data.label
                _tr_loss = _train(model, sents, labels, optimizer)

            train_loss.append(_tr_loss)

        for i, valid_data in enumerate(valid_iter):

            if model_name in ('BiLSTM_CNN_CRF', 'BiLSTM_CRF_Char'):
                sents = valid_data.word
                labels = valid_data.label
                chars = valid_data.char
                _vl_loss, _vl_acc = _valid(model, sents, labels, chars=chars)
            elif model_name == 'BiLSTM_CRF_Flair':
                sents = valid_data[0].word
                labels = valid_data[0].label
                flair_embs = valid_data[1]
                _vl_loss, _vl_acc = _valid(model, sents, labels, flair_embs=flair_embs)
            else:
                sents = valid_data.word
                labels = valid_data.label
                _vl_loss, _vl_acc = _valid(model, sents, labels)

            valid_loss.append(_vl_loss)
            valid_acc.append(_vl_acc)

        curr_f1_score = evaluate(model, valid_ds, word_vocab, label_vocab, use_flair=use_flair)

        scheduler.step(curr_f1_score)

        # deep copy the current best model
        if curr_f1_score > best_f1_score:
            best_model = copy.deepcopy(model.state_dict())
            best_f1_score = curr_f1_score

        writer.add_scalar('loss/train', np.mean(train_loss), epoch + 1)
        writer.add_scalar('loss/validation', np.mean(valid_loss), epoch + 1)
        writer.add_scalar('acc/validation', np.mean(valid_acc), epoch + 1)
        writer.add_scalar('f1/test', curr_f1_score, epoch + 1)


        total_train_loss.append(np.mean(train_loss))
        total_valid_loss.append(np.mean(valid_loss))
        total_valid_acc.append(np.mean(valid_acc))

        logger.debug('epoch {} / {}, train loss:{:.5f} valid loss:{:.5f} valid score:{:.3f}'.
                     format(epoch + 1, EPOCHS, total_train_loss[epoch],
                     total_valid_loss[epoch], curr_f1_score))

    # close tensorboardX file
    writer.close()

    # load best model
    model.load_state_dict(best_model)

    # save best model
    if save_model:
        torch.save(model.state_dict(), MODEL_FILE)
        logger.debug('trained model is saved to ', MODEL_FILE)

    if trial is not None:
        count = trial.number + 1
        print('{}/{} {}'.format(count, trial.n_trials, '*' * count))
        # evaluate model
        return evaluate(model,
                        valid_ds,
                        word_vocab,
                        label_vocab,
                        use_flair=use_flair)


    else:
        # evaluate model
        return evaluate(model,
                        test_ds,
                        word_vocab,
                        label_vocab,
                        each_f1_score,
                        is_final=True,
                        error_analysis=True,
                        use_flair=use_flair)

def load_model(word_vocab, label_vocab):

    EMBEDDING_DIM = word_vocab.vectors.size(1)
    bilstm_crf = BiLSTM_CRF(len(word_vocab),
                            label_vocab.stoi,
                            EMBEDDING_DIM,
                            HIDDEN_DIM, BATCH_SIZE)

    if CUDA:
        bilstm_crf = bilstm_crf.cuda()

    # load pre-trained parameters
    bilstm_crf.load_state_dict(
        torch.load(MODEL_FILE))

    return bilstm_crf

def valid(model_name,
          valid_file='',
          check_result_tag='',
          check_wrong_tag='',
          verbose=False):

    # logging initialize
    global logger
    logger = logging_init(verbose)

    # load vocabulary
    word_vocab = torch.load(VOCAB_FILE)
    label_vocab = torch.load(LABEL_FILE)

    # make dataset
    WORD = data.Field()
    LABEL = data.Field()
    # set vocab obj to xx.vocab
    setattr(WORD, "vocab", word_vocab)
    setattr(LABEL, "vocab", label_vocab)

    global VALID_FILE
    if valid_file:
        VALID_FILE = valid_file

    valid_ds = datasets.SequenceTaggingDataset(fields=[('word', WORD), ('label', LABEL)],
                                              path=VALID_FILE,
                                              separator=' '
                                              )

    logger.debug('validation file : {}'.format(VALID_FILE))
    logger.debug('num validation data : {}'.format(len(valid_ds)))

    # load model
    bilstm_crf = load_model(word_vocab, label_vocab)

    evaluate(bilstm_crf, valid_ds, word_vocab, label_vocab, is_final=True, error_analysis=True)

def get_uncertainty_score(sentence, uncertainty_method, verbose=False):
    pred_list = predict(model_name="", sentence=sentence, verbose=verbose, n_best=2)
    prob_list = np.asarray(np.array(pred_list)[:,1], dtype=np.float)
    score_list = np.asarray(np.array(pred_list)[:,2], dtype=np.float)
    if uncertainty_method == 'least_confident':
        uncertainty_score = 1 - np.prod(prob_list, axis=1)[0]
    elif uncertainty_method == 'margin_based':
        n_probs = np.prod(prob_list, axis=1)
        uncertainty_score = - (n_probs[0] - n_probs[1])
    else:
        x = softmax(np.sum(score_list, axis=1))  # re-normalized
        uncertainty_score = -np.sum(x * np.log(x))
    return uncertainty_score

def predict(model_name,
            predict_file='',
            out_file='',
            sentence='',
            save_file=False,
            verbose=False,
            target_label_name='',
            n_best=1,
            path_only=True,
            omission=True
            ):

    # set target file
    global PREDICT_FILE
    global OUT_FILE
    if predict_file:
        PREDICT_FILE = predict_file
    if out_file:
        OUT_FILE = out_file

    if verbose:
        print('model file: ', MODEL_FILE)
        if sentence:
            print('target sent: ', sentence)
        else:
            print('target file: ', PREDICT_FILE)

    # load vocabulary
    word_vocab = torch.load(VOCAB_FILE)
    label_vocab = torch.load(LABEL_FILE)

    # make dataset
    WORD = data.Field()
    LABEL = data.Field()
    # set vocab obj to xx.vocab
    setattr(WORD, "vocab", word_vocab)
    setattr(LABEL, "vocab", label_vocab)

    # load model
    bilstm_crf = load_model(word_vocab, label_vocab)

    if sentence:
        path_only = False
        bilstm_crf.eval()
        if path_only:
            with torch.no_grad():
                x = [word_vocab.stoi[word] for word in sentence.split(' ')]  # [words]
                x = torch.t(LongTensor(x).expand(1, -1))  # [words, 1]
                # pred, pred_prob, pred_all_prob = bilstm_crf.predict_prob(x, n_best=1)
                n_best_list = bilstm_crf.predict_prob(x, n_best=n_best)
            if target_label_name:
                print('target_label : ', target_label_name)
                target_label_idx_b = label_vocab.stoi['B-' + target_label_name]
                target_label_idx_i = label_vocab.stoi['I-' + target_label_name]
                target_idxes = np.where((np.array(pred[0])==target_label_idx_b)|(np.array(pred[0])==target_label_idx_i))[0]
                pred_all_prob = [np.array(pred_all_prob[0])[target_idxes, :]]
            pred_list = []
            for _n_best in n_best_list:
                path, prob, score = _n_best
                pred_list.append([sents_itos([path], label_vocab, omission=omission)[0], prob, score])
                if verbose:
                    print(pred_list[-1][0])
                    print(['{:.3f}'.format(n) for n in prob])
            return pred_list  # [path, prob, score]
        else:
            x = [word_vocab.stoi[word] for word in sentence.split(' ')]  # [words]
            x = torch.t(LongTensor(x).expand(1, -1))  # [words, 1]
            pred_indices = bilstm_crf.predict(x)
            pred_labels = sents_itos(pred_indices, label_vocab, omission=omission)[0]
            if verbose:
                print(pred_labels)
            return pred_labels

    else:
        pred_ds = datasets.SequenceTaggingDataset(fields=[('word', WORD)],
                                                  path=PREDICT_FILE,
                                                  separator=' '
                                                  )

        pred_iter = data.BucketIterator(pred_ds,
                                        batch_size=1,
                                        sort_key=lambda x: len(x.word),
                                        repeat=False,
                                        shuffle=False,
                                        device="cuda")


        pred_idxes = _predict(bilstm_crf, pred_iter, use_flair=False)
        pred_labels = sents_itos(pred_idxes, label_vocab)

        if save_file:
            docs = [pred_ds.examples[i].word for i in range(len(pred_ds))]
            sequence_to_csv(filname=OUT_FILE, docs=docs, pred_labels=pred_labels)
            print('predicted to : ', OUT_FILE)
