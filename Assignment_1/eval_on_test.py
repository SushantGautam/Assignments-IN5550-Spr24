#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import re
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPClassifier
# from datasets import load_dataset

# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# from sklearn.metrics import classification_report
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.model_selection import train_test_split

# import gensim
# from gensim.models import Word2Vec
# from gensim.models.callbacks import CallbackAny2Vec
# from gensim import corpora, models, similarities, downloader

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# torch.manual_seed(42)
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# import numpy as np
# np.random.seed(42)


# In[1]:


import re
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
trns = lambda s: " ".join([lemmatizer.lemmatize(word) for word in re.sub('[\W\d]+', ' ', s).lower().split() if word not in stop_words])


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.Module):
    def __init__(self, layer_map=[128], dropout_rate=0.5, merge_operation='XXX', embedding_type='xxx', oov_handler="xxx", freeze_emb=True, RNN_type='xxx', fusion_type='xxx', hidden_dim=None, num_layers=None, bidirectional=None):
        super().__init__()
        # Assume getWordWV, label2id are defined elsewhere
        embedding_wv = getWordWV(oov_handler, embedding_type)  
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_wv.vectors), freeze=freeze_emb)
        self.embedding_dim = embedding_wv.vector_size
        self.key_to_index = embedding_wv.key_to_index
        self.pad_index = embedding_wv.get_index("[PAD]")
        self.unk_index = embedding_wv.get_index("[UNK]")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.RNN_type = RNN_type.lower()
        self.merge_operation = merge_operation.lower()
        self.fusion_type = fusion_type.lower()
        
        rnn_input_size = self.embedding_dim
        rnn_layer = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}.get(self.RNN_type)
        if rnn_layer:
            self.rnn = rnn_layer(rnn_input_size, self.hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"Unsupported RNN_type: {RNN_type}")

        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        for size in layer_map:
            self.fc_layers.append(nn.Linear(fc_input_dim, size))  # Adjusting input size for FC layers
            self.hidden_dim = size  # Updating hidden_dim to match the output of the current layer
            fc_input_dim = size
        
        self.fc_out = nn.Linear(self.hidden_dim, len(label2id))  # Assuming label2id is defined elsewhere


    def forward(self, x, lengths):
        batch_size = x.size(0)
        embedded = self.embedding(x)

        # Pack the padded sequence
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Process with RNN
        packed_rnn_out, _ = self.rnn(packed_embedded)
        
        # If you need the output for each sequence, unpack the sequence
        # rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
        
        # Handling different fusion types
        if self.fusion_type == 'last':
            if self.RNN_type == 'LSTM':
                num_directions = 2 if self.rnn.bidirectional else 1
                if num_directions == 2:
                    # concatenate the hidden states from the last layer of both directions
                    h_n_last_layer = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
                else:
                    # take the last layer's hidden state
                    h_n_last_layer = h_n[-1, :, :]
                    rnn_out = h_n_last_layer
            else:
                rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
                rnn_out = rnn_out[torch.arange(batch_size), lengths - 1]
        elif self.fusion_type == 'first':
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
            rnn_out = rnn_out[:, 0, :]
        # Add or modify fusion types as needed
        elif self.fusion_type == 'add':
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
            rnn_out = rnn_out[:, 0, :] + rnn_out[torch.arange(batch_size), lengths - 1]
        elif self.fusion_type == 'max':
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
            rnn_out, _ = torch.max(rnn_out, dim=1)
        elif self.fusion_type == 'sum':
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
            rnn_out = torch.sum(rnn_out, dim=1)
        elif self.fusion_type == 'average':
            rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=True)
            rnn_out = torch.mean(rnn_out, dim=1)
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

        rnn_out = rnn_out
        for layer in self.fc_layers:
            rnn_out = F.relu(layer(rnn_out))
            rnn_out = self.dropout(rnn_out)

        output = self.fc_out(rnn_out)
        return output

# In[3]:


class CustomDataset(Dataset):
    def __init__(self, sentences, labels, word_emb_model, unk_index, _label2id):
        self.unk_index = unk_index
        self.label2id =  _label2id
        self.tokens = [
            [
                word_emb_model.get(token.lower(), self.unk_index)
                for token in sentence.split()
            ] 
            for sentence in sentences
        ]

        unk_tokens = sum(token == self.unk_index for document in self.tokens for token in document)
        n_tokens = sum(len(document) for document in self.tokens)
        print(f"Percentage of unknown tokens: {unk_tokens / n_tokens * 100.0:.2f}%")
        self.label = list(labels)


    def __getitem__(self, index):
        current_tokens = self.tokens[index]
        current_label = self.label[index]

        x = torch.LongTensor(current_tokens)
        y = torch.LongTensor([self.label2id[current_label]])
        return x, y

    def __len__(self):
        return len(self.tokens)


class CollateFunctor:
    def __init__(self, padding_index: int, max_length: int):
        self.padding_index = padding_index
        self.max_length = max_length

    # Gets a list of outputs from the dataset and outputs tensors
    def __call__(self, samples):
        input_ids = [x for x, _ in samples]
        labels = [y for _, y in samples]
        lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first = True,
            padding_value=self.padding_index, 
        )
        input_ids_padded = input_ids_padded[:, :self.max_length]
        return input_ids_padded, lengths.long(), torch.LongTensor(labels)

def evaluate(model, data_loader,_id2label ):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, lengths, labels in data_loader:
            inputs = inputs.squeeze(1)
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=list(_id2label.values()), zero_division=0))


# In[4]:


import argparse
def main(csv_path, model_path):
    model = torch.load(model_path)

    print(model)
    dataset = pd.read_csv(csv_path, compression="gzip")
    dataset['abstract_org'] = dataset['abstract'] #backup
    dataset['abstract'] = dataset['abstract'].apply(trns)
    X_test, y_test = dataset['abstract'].values, dataset['label'].values

    id2label = dict(enumerate(dataset['label'].unique()))
    label2id = {v: k for k, v in id2label.items()} 
    
    test_loader = DataLoader(CustomDataset(X_test, y_test, model.key_to_index, model.unk_index, _label2id=label2id ), batch_size=10000, shuffle=False, drop_last=False, 
                                collate_fn=CollateFunctor(model.pad_index, max_length=400))


    evaluate(model, test_loader, _id2label=id2label)


# In[5]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="arxiv_train.csv.gz", help="Path to the .csv.gz file")
    parser.add_argument("--model", type=str, default="torch_demo.bin", help="Path to the .bin trained model file")
    args = parser.parse_args()
    main(args.test, args.model)


# In[8]:


# input_strings = ["Quantum state readout is a key component of quantum technologies,.", "Bible said the sun and moon revolve around the earth"]

# model = torch.load("torch_demo.bin")
# id2label = model.id2label
# label2id = {v: k for k, v in model.id2label.items()} 
# print(model)
# tokenized_sequences = [[model.key_to_index[token] if token in model.key_to_index else model.unk_index for token in trns(string).split()] for string in input_strings]
# max_token_index = max(len(seq) for seq in tokenized_sequences)
# padded_sequences = [seq + [model.pad_index] * (max_token_index - len(seq)) for seq in tokenized_sequences]
# probabilities = torch.softmax(model(torch.tensor(padded_sequences)), dim=1)* 100
# labels_with_percentage = [[(model.id2label[idx], f"{percentage:.2f}%") for idx, percentage in enumerate(row)] for row in probabilities]
# [max(row, key=lambda x: float(x[1][:-1])) for row in labels_with_percentage]


# In[ ]:


# jupyter nbconvert eval_on_test.ipynb --to python


# In[ ]:


##### Final Model test script:
# python eval_on_test.py --test /fp/projects01/ec30/IN5550/obligatories/1/arxiv_train.csv.gz --model /fp/projects01/ec30/sushant/torch_demo.bin
# python eval_on_test.py --test arxiv_train.csv.gz --model torch_demo.bin


# In[ ]:





# In[ ]:





# In[ ]:




