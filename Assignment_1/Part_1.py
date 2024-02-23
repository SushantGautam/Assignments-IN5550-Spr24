#!/usr/bin/env python
# coding: utf-8

# # Load dependencies:

# In[21]:


import re
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(42)

import numpy as np
np.random.seed(42)

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
transformation = lambda s: " ".join([lemmatizer.lemmatize(word) for word in re.sub('[\W\d]+', ' ', s).lower().split() if word not in stop_words])


# # Data preparation

# In[22]:


# read /fp/projects01/ec30/IN5550/obligatories/1/arxiv_train.csv.gz
dataset = pd.read_csv("/fp/projects01/ec30/IN5550/obligatories/1/arxiv_train.csv.gz", compression="gzip")
dataset['abstract_org'] = dataset['abstract'] #backup
# dataset['abstract'] = dataset['abstract'].str.replace('\n', ' ')
# dataset['abstract'] = dataset['abstract'].str.replace('\t', ' ')
# dataset['abstract'] = dataset['abstract'].str.replace('_', ' ')
# dataset['abstract'] = dataset['abstract'].str.replace('-', ' ')
# dataset['abstract'] = dataset['abstract'].str.replace('\d+', ' ')
# dataset['abstract'] = dataset['abstract'].apply(lambda x: x.lower().split(" "))
# dataset['abstract'] = dataset['abstract'].apply(lambda x: [w for w in x if not w in stop_words])
# dataset['abstract'] = dataset['abstract'].apply(lambda x: " ".join([lemmatizer.lemmatize(y) for y in x]))
dataset['abstract'] = dataset['abstract'].apply(transformation)
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset = dataset.reset_index(drop=True)
dataset['id']= dataset.index


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(dataset['id'], dataset['label'], test_size=0.3, random_state=42, stratify=dataset['label'])
id2label = dict(enumerate(dataset['label'].unique()))
label2id = {v: k for k, v in id2label.items()}


# In[25]:


corpus = dataset['abstract']

count_vectorizer = CountVectorizer()
X_count = count_vectorizer.fit_transform(corpus)
# count_vectorizer.get_feature_names_out()

binary_vectorizer = CountVectorizer(binary=True)
X_binary = binary_vectorizer.fit_transform(corpus)

tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus)

print("X- shpe", X_tfidf.shape)


# # Helper functions

# In[26]:


class CustomDataset(Dataset):
    def __init__(self, indices, X_vec, labels):
        self.indices = list(indices)
        self.X_vec = X_vec
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        # Convert sparse matrix slice to dense
        x = self.X_vec[index].toarray()
        y_str = self.labels[index]
        y= label2id[y_str]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

num_classes = len(id2label)


# In[83]:


class TextClassifier(nn.Module):
    def __init__(self, input_size, layer_map=[128], dropout_rate=0.6):
        super(TextClassifier, self).__init__()
        self.fc_layers = nn.ModuleList()  # ModuleList to hold dynamically created layers
        self.dropout = nn.Dropout(dropout_rate)
        prev_layer_size = input_size
        for size in layer_map:
            self.fc_layers.append(nn.Linear(prev_layer_size, size))
            prev_layer_size = size
        self.fc_out = nn.Linear(prev_layer_size, num_classes)

    def forward(self, x):
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        return x


# In[84]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# In[85]:


def train(model, criterion, optimizer, train_loader, val_loader, epochs=5):
    model.to(device)
    train_loss, val_loss = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.squeeze(1).to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
        print(f'Epoch {epoch+1}, Train Loss: {train_loss[-1]:.6f}, Val. Loss: {val_loss[-1]:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, F1: {f1:.6f}')

    return train_loss, val_loss


# In[86]:


lr=0.0005
batch_size=10000
epochs=50
criterion = nn.CrossEntropyLoss()


# In[87]:


def train_evaluate(X_vector, model_layer_map=[128], lr_=lr, epochs_=epochs, batch_size_=batch_size ): # custom model can also be passes
    print("Initializing model with lr = {}, batch_size = {}, epochs = {}".format(lr_, batch_size_, epochs_))
    model=TextClassifier(layer_map=model_layer_map, input_size=X_vector.shape[1])
    optimizer= torch.optim.Adam(model.parameters(), lr=lr_)
    train_loader = DataLoader(CustomDataset(X_train, X_vector, y_train), batch_size=batch_size_, shuffle=True)
    test_loader = DataLoader(CustomDataset(X_test, X_vector, y_test), batch_size=batch_size_, shuffle=False)
    train_loss, val_loss = train(model, criterion, optimizer, train_loader, test_loader, epochs=epochs_)
    evaluate(model, test_loader)
    return model, train_loss, val_loss


# # Experiments

# ## BOW Variations

# In[16]:


print("Experiment A-1: Using Binary vectorizer")
model_b, train_loss_b, val_loss_b = train_evaluate(X_binary, lr_=lr)
plot_loss_curve(train_loss_b, val_loss_b)


# In[31]:


print("Experiment A-2: Using Count vectorizer")
model_c, train_loss_c, val_loss_c = train_evaluate(X_count, lr_=lr)
plot_loss_curve(train_loss_c, val_loss_c)


# In[32]:


print("Experiment A-3: Using TF-IDF vectorizer")
model_t, train_loss_t, val_loss_t = train_evaluate(X_tfidf, lr_=0.001) #lower rate for IFIDF
plot_loss_curve(train_loss_t, val_loss_t)


# ## Layer Variations with best vectorizer (TF-IDF )
# Note: Size of a single hidden layer used above for experiment A-3 was [128].

# In[103]:


lr= 0.0005
batch_size=10000
epochs=50


# In[51]:


layer_map_ =[256]
print("Experiment B-1: Using hidden layers of size", layer_map_ )
model_l1, train_loss_l1, val_loss_l1 = train_evaluate(X_tfidf, model_layer_map= layer_map_, lr_=lr, epochs_=epochs, batch_size_=batch_size)
plot_loss_curve(train_loss_l1, val_loss_l1)


# In[91]:


layer_map_ =[4096]
print("Experiment B-2: Using hidden layers of size", layer_map_ )
model_l2, train_loss_l2, val_loss_l2 = train_evaluate(X_tfidf, model_layer_map= layer_map_, lr_=lr, epochs_=epochs, batch_size_=batch_size)
plot_loss_curve(train_loss_l2, val_loss_l2)


# In[92]:


layer_map_ =[4096, 128]
print("Experiment B-3: Using hidden layers of size", layer_map_ )
model_l3, train_loss_l3, val_loss_l3 = train_evaluate(X_tfidf, model_layer_map= layer_map_, lr_=lr, epochs_=epochs, batch_size_=batch_size)
plot_loss_curve(train_loss_l3, val_loss_l3)


# In[105]:


layer_map_ =[512, 128, 16]
print("Experiment B-4: Using hidden layers of size", layer_map_ )
model_l4, train_loss_l4, val_loss_l4 = train_evaluate(X_tfidf, model_layer_map= layer_map_, lr_=lr, epochs_=epochs, batch_size_=batch_size)
plot_loss_curve(train_loss_l4, val_loss_l4)


# # Inference using best model using its vectorizer

# In[101]:


best_model = model_t
best_tokenizer = tfidf_vectorizer

input_text = "They believed sun and moon revolved around the earth circle parameter square, what a dsdddsdhbsdjnsd"
vector_text = tfidf_vectorizer.transform([input_text]).toarray()
probabilities = best_model(torch.tensor(vector_text, dtype=torch.float32).to(device))[0]

sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
_ = [print(f"Label: {id2label[index.item()]}, Probability: {prob.item()}") for index, prob in zip(sorted_indices, sorted_probs)]

