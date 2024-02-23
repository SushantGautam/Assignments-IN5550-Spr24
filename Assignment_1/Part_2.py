#!/usr/bin/env python
# coding: utf-8

# ## Load necessary libraries and prepare dataset

# In[ ]:


import re
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from datasets import load_dataset

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import gensim
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim import corpora, models, similarities, downloader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

import numpy as np
np.random.seed(42)


# In[ ]:


nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
transformation = lambda s: " ".join([lemmatizer.lemmatize(word) for word in re.sub('[\W\d]+', ' ', s).lower().split() if word not in stop_words])

# read /fp/projects01/ec30/IN5550/obligatories/1/arxiv_train.csv.gz
dataset = pd.read_csv("/fp/projects01/ec30/IN5550/obligatories/1/arxiv_train.csv.gz", compression="gzip")
dataset['abstract_org'] = dataset['abstract'] #backup
dataset['abstract'] = dataset['abstract'].apply(transformation)


# ## Train custom word embedding model

# In[ ]:


print("Available pre-trained corpora:", list(downloader.info()['corpora'].keys()))
print([folder for folder in os.listdir('/fp/projects01/ec30/corpora')])


# In[ ]:


# dataset_arxiv_hf_df = pd.DataFrame({'abstract': load_dataset("gfissore/arxiv-abstracts-2021")["train"]['abstract']})
# dataset_arxiv_hf_df['abstract'] = dataset_arxiv_hf_df['abstract'].apply(transformation)
# concatenated_df = pd.concat([pd.DataFrame({'abstract': dataset['abstract'].values}), dataset_arxiv_hf_df], ignore_index=True)
# concatenated_df_ = concatenated_df.dropna(subset=['abstract'])
# concatenated_df_.to_csv('corpus.csv', index=False)
# corpus_to_train = concatenated_df['abstract']

# corpus_to_train = pd.read_csv('corpus.csv').dropna()['abstract']


# In[ ]:


# #number of abstracts in corpus_to_train : 2,079,484, from dataset_arxiv_hf: 2M  from our dataset['abstract']: 80K
# print("Total number of tokens:", corpus_to_train.apply(lambda x: len(x.split())).sum()) # 179,355,863
# print("Number of unique tokens:", len(set(" ".join(corpus_to_train).split()))) # 471,322


# In[ ]:


class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch > 0:
            current_loss = loss - self.prev_loss
            self.losses.append(current_loss)
            print(f'Epoch {self.epoch}: Loss: {current_loss}')
        else:
            self.losses.append(loss)
            print(f'Epoch {self.epoch}: Loss: {loss}')
        self.prev_loss = loss
        self.epoch += 1


# In[ ]:


# corpus_gn = corpus_to_train.apply(lambda x: x.split()).values[:10]
# model_gn = Word2Vec(vector_size=300, window=5, min_count=5, workers=60, epochs=20, sg=1)
# model_gn.build_vocab(corpus_iterable=corpus_gn)
# print('\n'.join([f"{member}: {size / (1024 * 1024):.2f} MB" for member, size in model_gn.estimate_memory().items()]))


# In[ ]:


# loss_logger = LossLogger()
# model_gn.train(corpus_gn, total_examples=len(corpus_gn), epochs=model_gn.epochs, compute_loss=True, callbacks=[loss_logger])
# loss_logger.losses = [28249280.0, 16752928.0, 14551592.0, 8131464.0, 1066216.0, 1044392.0, 1033128.0, 1092200.0, 1011840.0, 979288.0, 932632.0, 941664.0, 908408.0, 856944.0, 844160.0, 841848.0, 884648.0, 811488.0, 818648.0, 762072.0]
# print(loss_logger.losses)
# plt.plot(loss_logger.losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss per Epoch of out custom word embedding model')
# plt.show()
# model_gn.wv.vectors.shape #model.wv.index_to_key


# In[ ]:


# model.save("xword2vec_300_5_5_20_1")


# ## Choose and load Pretrained models

# In[ ]:


print("Available pre-trained modules:", list(downloader.info()['models'].keys()))


# In[ ]:


model_list = {
            "glove-twitter-25": downloader.load("glove-twitter-25"),
            "word2vec-google-news-300": downloader.load("word2vec-google-news-300"),
            "glove-wiki-gigaword-300": downloader.load("glove-wiki-gigaword-300"),
            "fasttext-wiki-news-subwords-300": downloader.load("fasttext-wiki-news-subwords-300"),
        }


# In[ ]:


# all_words = [word for abstract in dataset['abstract'] for word in abstract.split()]
# unique_words = set(all_words)
# print("Total Words in our corpus:", len(all_words))
# print("Unique words in our corpus:", len(unique_words))

# for model_name in ['glove-twitter-25', 'word2vec-google-news-300', 'glove-wiki-gigaword-300', 'fasttext-wiki-news-subwords-300']:
#     word_model = model_list[model_name]
#     words_in_model = [word for word in all_words if word in word_model]
#     unique_words_in_model =  set(words_in_model)
#     print("\n>> {} Model loaded.".format(model_name))
#     print(f"Unique words from our corpus present in model: {len(unique_words_in_model)} ({(len(unique_words_in_model) / len(unique_words)) * 100:.2f}%)")
#     print(f"Total words from our corpus present in model: {len(words_in_model)} ({(len(words_in_model) / len(all_words)) * 100:.2f}%)")
#     print("Total number of vocabulary words in model:", len(word_model.index_to_key))


# - Total Words in our corpus: 7950716
# -  Unique words in our corpus: 85756
# 
# **GloVe Twitter 25 Model:**
# - Unique words from our corpus present in model: 36,883 (43.01%)
# - Total words from our corpus present in model: 7,360,299 (92.57%)
# - Total number of vocabulary words in model: 1,193,514
# 
# **Word2Vec Google News 300 Model:**
# - Unique words from our corpus present in model: 34,500 (40.23%)
# - Total words from our corpus present in model: 7,501,847 (94.35%)
# - Total number of vocabulary words in model: 3,000,000
# 
# **GloVe Wiki Gigaword 300 Model:**
# - Unique words from our corpus present in model: 44,642 (52.06%)
# - Total words from our corpus present in model: 7,694,945 (96.78%)
# - Total number of vocabulary words in model: 400,000
# 
# **FastText Wiki News Subwords 300 Model:**
# - Unique words from our corpus present in model: 40,637 (47.39%)
# - Total words from our corpus present in model: 7,664,711 (96.40%)
# - Total number of vocabulary words in model: 999,999
# 

# ## Define models and dataloader

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dataset['abstract'].values, dataset['label'].values, test_size=0.3, random_state=42, stratify=dataset['label'])
id2label = dict(enumerate(dataset['label'].unique()))
label2id = {v: k for k, v in id2label.items()}


# In[ ]:


class CustomDataset(Dataset):
    def __init__(self, sentences, labels, word_emb_model, unk_index):
        self.unk_index = unk_index
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
        y = torch.LongTensor([label2id[current_label]])
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

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first = True,
            padding_value=self.padding_index,
        )
        input_ids_padded = input_ids_padded[:, :self.max_length]
        return input_ids_padded, torch.LongTensor(labels)

num_classes = len(id2label)


# In[ ]:


def getWordWV(oov_handler="mean", model_name="word2vec_300_5_5_20_1"):
    if "custom_" in model_name:
        word_model = gensim.models.KeyedVectors.load(model_name.replace("custom_", "")).wv
    else:
        word_model = model_list[model_name]

    if oov_handler =="mean":
        word_model["[UNK]"] = torch.tensor(word_model.vectors).mean(dim=0).numpy() #average
    elif oov_handler =="random":
        word_model["[UNK]"] = np.random.uniform(low=-0.3, high=0.3, size=(word_model.vector_size,)) # random''
    else:
        word_model["[UNK]"] = torch.zeros(word_model.vector_size).numpy()

    word_model["[PAD]"] = torch.zeros(word_model.vector_size).numpy()
    return word_model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassifier(nn.Module):
    def __init__(self, layer_map=[128], dropout_rate=0.5, merge_operation='average', embedding_type='glove-twitter-25', oov_handler="mean", freeze_emb=True):
        super().__init__()
        embedding_wv = getWordWV(oov_handler, embedding_type) # return model.wv
        self.key_to_index = embedding_wv.key_to_index
        self.pad_index = embedding_wv.get_index("[PAD]")
        self.unk_index = embedding_wv.get_index("[UNK]")
        self.embedding_dim = embedding_wv.vector_size
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_wv.vectors),  freeze=freeze_emb)
        self.merge_operation = merge_operation
        self.fc_layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        self.id2label = id2label
        prev_layer_size = self.embedding_dim  # Adjusted to use embedding_dim as input size
        if self.merge_operation == "concat":
            prev_layer_size= prev_layer_size *3
        for size in layer_map:
            self.fc_layers.append(nn.Linear(prev_layer_size, size))
            prev_layer_size = size
        self.fc_out = nn.Linear(prev_layer_size, len(label2id))

    def forward(self, x):
        # x is assumed to be a batch of sentences with token indices
        embedded = self.embedding(x)  # Convert token indices to embeddings

        if self.merge_operation == 'average':
            sentence_lengths = (x != 0).sum(dim=1, keepdim=True)  # Assuming padding index is 0
            sentence_embeddings = embedded.sum(dim=1) / sentence_lengths

        elif self.merge_operation == 'sum':
            sentence_embeddings = embedded.sum(dim=1)

        elif self.merge_operation == 'max':
            sentence_embeddings, _ = embedded.max(dim=1)

        elif self.merge_operation == 'concat':
            # Assuming we want to concatenate average, sum, and max embeddings for a richer representation
            avg_embeddings = embedded.sum(dim=1) / (x != 0).sum(dim=1, keepdim=True)
            sum_embeddings = embedded.sum(dim=1)
            max_embeddings, _ = embedded.max(dim=1)
            sentence_embeddings = torch.cat((avg_embeddings, sum_embeddings, max_embeddings), dim=1)

        else:
            raise ValueError(f"Unsupported merge_operation: {self.merge_operation}")


        for layer in self.fc_layers:
            sentence_embeddings = F.relu(layer(sentence_embeddings))
            sentence_embeddings = self.dropout(sentence_embeddings)
        output = self.fc_out(sentence_embeddings)
        return output


# In[ ]:


def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.squeeze(1).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=list(id2label.values()), zero_division=0))


def plot_loss_curve(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[ ]:


def train(model, criterion, _optimizer, train_loader, val_loader, epochs, verbose):
    model.to(device)
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.squeeze(1).to(device), labels.to(device)

            _optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            _optimizer.step()
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.squeeze(1).to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation loss
        val_loss.append(running_loss / len(val_loader))
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        if verbose:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.6f}, Val. Loss: {val_loss[-1]:.6f}, F1: {f1:.6f}')
                        # Precision: {precision:.6f}, Recall: {recall:.6f},
        else:
            print(f'{f1*100:.2f},', end=" ")
    print("")
    return train_loss, val_loss


# In[ ]:


max_length = 280 #260
criterion = nn.CrossEntropyLoss()
def train_evaluate(merge_operation, embedding_type,  oov_handler, freeze_emb,
                   model_layer_map, lr_, epochs_, batch_size_, verbose_=True): # custom model can also be passes
    print("Model with merge_operation = {}, embedding_type: {}, oov_handle: {}, freeze_emb: {}, model_layer_map: {}, lr: {}, batch_size: {}".format(merge_operation, embedding_type, oov_handler, freeze_emb, model_layer_map, lr_, batch_size_))

    model=TextClassifier(layer_map=model_layer_map, merge_operation=merge_operation, embedding_type=embedding_type, oov_handler=oov_handler, freeze_emb=freeze_emb,)
    print("Trainable paramaters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_loader = DataLoader(CustomDataset(X_train, y_train, model.key_to_index, model.unk_index), batch_size=batch_size_, shuffle=True, drop_last=False,
                                 collate_fn= CollateFunctor(model.pad_index, max_length))
    test_loader = DataLoader(CustomDataset(X_test, y_test, model.key_to_index, model.unk_index), batch_size=batch_size_, shuffle=False, drop_last=False,
                                collate_fn= CollateFunctor(model.pad_index, max_length))

    print(model)
    optimizer= torch.optim.Adam(model.parameters(), lr=lr_)
    train_loss, val_loss = train(model, criterion, optimizer, train_loader, test_loader, epochs=epochs_, verbose=verbose_)
    evaluate(model, test_loader)
    return model, train_loss, val_loss


# # Experiments

# ##### Arguments for experimentations
# - merge_operation' average, sum, max, conact
# -  embedding_type (pretrained) = "glove-twitter-25", "word2vec-google-news-300", "glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300"
# -  embedding_type (our models) = "custom_word2vec_100_5_5_10_1", "custom_word2vec_300_5_5_20_1", "custom_fasttext_100_5_5_10_1"
# - oov_handler= mean, random, zero
# - freeze_emb= True, False

# In[ ]:


lr=0.01; batch_size=10000; epochs=100


# ## A: Different ways of merging the sequence of embeddings into a fixed-size

# In[ ]:


print("Experiment A-1: merge_operation-average")
_, train_loss, val_loss = train_evaluate(merge_operation='average', embedding_type='custom_word2vec_300_5_5_20_1', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[64], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment A-2: merge_operation-max")
_, train_loss, val_loss = train_evaluate(merge_operation='max', embedding_type='custom_word2vec_300_5_5_20_1', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment A-3: merge_operation-sum")
_, train_loss, val_loss = train_evaluate(merge_operation='sum', embedding_type='custom_word2vec_300_5_5_20_1', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment A-4: merge_operation-concat")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='custom_word2vec_300_5_5_20_1', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# ## B: Different ways of dealing with out-of-vocabulary words;

# In[ ]:


print("Experiment B-1: OOV handling-mean")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='glove-twitter-25', oov_handler="mean", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment B-2: OOV handling-random")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='glove-twitter-25', oov_handler="random", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# ## C: Different pretrained embedding models
# 
# *   List item
# *   List item
# 
# 

# In[ ]:


# "glove-twitter-25", "word2vec-google-news-300", "glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300"
print("Experiment C-1: embedding_type-glove-twitter-25")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='glove-twitter-25', oov_handler="zero", freeze_emb=True,
                  model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment C-2: word2vec-google-news-300")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='word2vec-google-news-300', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


print("Experiment C-3: glove-wiki-gigaword-300")
_, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='glove-wiki-gigaword-300', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[512], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# ## D: Pretrained vs custom models

# In[ ]:


print("Experiment D-1:")
_, train_loss_b, val_loss_b = train_evaluate(merge_operation='concat', embedding_type='custom_fasttext_100_5_5_10_1', oov_handler="zero", freeze_emb=True,
                   model_layer_map=[256], lr_=lr, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# ## E: Frozen or finetune embeddings layer

# In[ ]:


print("Experiment E-1: finetune word embeddings layer")
modelE, train_loss, val_loss = train_evaluate(merge_operation='concat', embedding_type='custom_word2vec_300_5_5_20_1', oov_handler="zero", freeze_emb=False,
                   model_layer_map=[512], lr_=0.0001, epochs_=epochs, batch_size_=batch_size, verbose_=False)
plot_loss_curve(train_loss[5:], val_loss[5:])


# In[ ]:


# torch.save(modelE.to('cpu'), "torch_demo.bin")


# In[ ]:





# In[ ]:




