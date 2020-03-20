# BERT encodeing 
import torch
import torch.nn.utils.rnn as rnn
from pytorch_transformers import BertTokenizer, BertModel, BertConfig

class BERT_Encoder(object):
    def __init__(self, bert_path):
        self.config = BertConfig.from_json_file(bert_path + 'config.json')
        self.bert_model = BertModel.from_pretrained(bert_path + 'pytorch_model.bin', config=self.config)
        self.tokenizer = BertTokenizer(bert_path + 'vocab.txt', do_lower_case=False, do_basic_tokenize=False)
        self.max_seq_length = 128
        self.pad_token = 0
        self.label2idx = None
        self.idx2label = None

    def make_labelmap(self, label_vocab):
        # label_list = ['[PAD]'] + list(label_vocab) + ['[CLS]'] + ['X']
        label_list = ['[PAD]'] + list(label_vocab) + ['[CLS]']
        self.label2idx = {label : i for i, label in enumerate(label_list)}
        self.idx2label = {i : label for label, i in self.label2idx.items()}

    def make_bert_dataset(self, dataset):
        tokens_list = []
        starts_list = []
        labels_list = []
        masks_list = []
        examples = dataset.examples
        for example in examples:
            tokens, starts, labels, masks = self.make_feature(example)
            tokens_list.append(tokens)
            starts_list.append(starts)
            labels_list.append(labels)
            masks_list.append(masks)
        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long)
        starts_tensor = torch.tensor(starts_list, dtype=torch.uint8)
        # Zero-pad up to the max label length.
        max_label_length = max(map(lambda labels: len(labels), labels_list))
        padding_length = self.max_label_length - len(tokens_ids)
        labels_list = labels_list + ([self.pad_token] * padding_length)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        masks_tensor = torch.tensor(masks_list, dtype=torch.uint8)
        return torch.utils.data.TensorDataset(tokens_tensor, starts_tensor, labels_tensor, masks_tensor)

    def make_feature(self, example):
        tokens = []
        starts_ids = []
        labels = []
        for word, label in zip(example.word, example.label):
            _tokens = self.tokenizer.tokenize(word)
            labels.append(label)
            for i, _token in enumerate(_tokens):
                tokens.append(_token)
                if i == 0:
                    starts_ids.append(1)
                    # labels.append(label)
                else:
                    starts_ids.append(0)
                    # labels.append("X")

        tokens = ['[CLS]'] + tokens
        starts_ids = [0] + starts_ids
        labels = ['[CLS]'] + labels
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        labels_ids = [self.label2idx[label] for label in labels] 

        # Trim by max_seq_length
        if len(tokens_ids) > self.max_seq_length:
            tokens_ids = tokens_ids[:self.max_seq_length]
            starts_ids = starts_ids[:self.max_seq_length]
        
        # create masks
        masks_ids = [1] * len(labels_ids)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(tokens_ids)
        tokens_ids = tokens_ids + ([self.pad_token] * padding_length)
        starts_ids = starts_ids + ([self.pad_token] * padding_length)
        return tokens_ids, starts_ids, labels_ids, masks_ids

    def encode(self, input_features):
        # input_features : batch * max_seq_length
        self.bert_model.eval()
        with torch.no_grad():
            bert_features = self.bert_model(input_features)[0]  # batch * max_seq_length * hidden
        # use the representation of the first sub-token
        return bert_features

    def extract_first_subtokens(self, input_features, labels_ids):
        pass

