from torch.utils.data import Dataset, DataLoader
import torch
from functools import partial
import sentencepiece
import pandas as pd
import gzip
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from functools import partial


device = "cuda" if torch.cuda.is_available() else "cpu"
id2label = {0: 'O',1: 'I-ORG', 2: 'I-PER', 3: 'I-LOC', 4: 'B-ORG', 5: 'B-PER', 6: 'B-LOC'}
label2id = {v:i for i,v in id2label.items()}
def read_gzipped_tsv(filepath):
    df =pd.read_csv(filepath, sep='\t', header=None, names=['Word', 'Type'], keep_default_na=True, na_filter=True,skip_blank_lines=False)
    return df[:100]

# Function to process the DataFrame and group words/types by sentences
def process_data(df):
    # Initialize lists to hold sentences and their types
    sentences = []
    sentence_types = []
    
    current_sentence = []
    current_types = []
    
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        # Check for a sentence separator (empty row)
        if pd.isna(row['Word']):
            # if current_sentence:  # If the current sentence list is not empty
            sentences.append(current_sentence)
            sentence_types.append(current_types)
            current_sentence = []  # Reset for the next sentence
            current_types = []
        else:
            current_sentence.append(row['Word'])
            current_types.append(row['Type'])
    
    # Don't forget to add the last sentence if the file doesn't end with an empty row
    if current_sentence:
        sentences.append(current_sentence)
        sentence_types.append(current_types)
    
    return pd.DataFrame({'Words': sentences, 'Types': sentence_types})


def read_and_combine_datasets(fpaths):
    """
    Read and combine datasets from given file paths.

    Args:
        fpaths (list of str): List of file paths.

    Returns:
        combined_df (DataFrame): Combined dataset.
    """
    combined_data = []
    for fpath in fpaths:
        temp_df = read_gzipped_tsv(fpath)
        processed_df = process_data(temp_df)
        combined_data.append(processed_df)

    sentences_df = pd.concat(combined_data, ignore_index=True)

    # Rename columns to match the desired output
    sentences_df.columns = ['sentence', 'label']
    # sentences_df = sentences_df[sentences_df['sentence'].apply(len) >= 2]

    return sentences_df


def stratified_split_train_set(train_df, test_size=0.2, random_state=42):
    # Create bins for stratification
    bins = [0, 5, 10, 15, 20, 25, 30, 55, 80, 100, 200]
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-55', '50-80', '80-100', '100-200']
    
    # Apply binning
    train_df['strat_bin'] = pd.cut(train_df["sentence"].apply(len), bins=bins, labels=labels)
    
    # Check if any bin has fewer than 2 samples
    bin_counts = train_df['strat_bin'].value_counts()
    valid_bins = bin_counts[bin_counts >= 2].index.tolist()
    
    # Filter the dataset to include only rows with valid bins
    valid_df = train_df[train_df['strat_bin'].isin(valid_bins)]
    
    if len(valid_df) < len(train_df):
        print(f"Warning: {len(train_df) - len(valid_df)} rows excluded due to sparse bins.")
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        valid_df["sentence"],
        valid_df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=valid_df['strat_bin']
    )
    
    return X_train, X_test, y_train, y_test

def analyze_dataset(dataset, labels):
    num_examples = len(dataset)
    num_tokens = sum(len(text) for text in dataset)
    named_entities = dict(Counter(ent for label in labels for ent in label).most_common())
    total_unique_words = len(set(word for sentence in dataset for word in sentence))
    return "Len: "+ str(num_examples),  "Tokens: "+str(num_tokens), "Unique: "+str(total_unique_words), "Entities: "+ str(named_entities)


def combine_and_split_datasets(train_langs=["en", "it", "af", "sw", "de"], data_dir="/fp/projects01/ec30/IN5550/obligatories/2/data", test_size=0.2):
    """
    Combine, split datasets based on specified languages, and prepare training and validation sets.

    Args:
        train_langs (list of str): Languages for training data.
        val_langs (list of str): Languages for validation data.
        data_dir (str): Directory where the datasets are stored.

    Returns:
        training_set, X_train, X_test, y_train, y_test, validation_set: Prepared datasets.
    """
    # Combine training datasets
    train_paths = [f"{data_dir}/train-{lang}.tsv.gz" for lang in train_langs]
    training_set = read_and_combine_datasets(train_paths)

    # Stratified split on training set
    _X_train, _X_val, _y_train, _y_val = stratified_split_train_set(training_set, test_size=test_size)
    print("Train Language(s):", train_langs)
    print("Train Set:", analyze_dataset(_X_train,_y_train ))
    print("Val Set:",  analyze_dataset(_X_val, _y_val))

    return list(_X_train), list(_X_val) , list(_y_train), list(_y_val)

def getTestDatasets(val_langs=["en"], data_dir="/fp/projects01/ec30/IN5550/obligatories/2/data"):
    all_set ={}
    for lang in val_langs:
        val_set = read_and_combine_datasets([f"{data_dir}/dev-{lang}.tsv.gz"])
        _X_test,_y_test  = list(val_set["sentence"]), list(val_set["label"])
        print(f"Test Set({lang}):",  analyze_dataset(_X_test,_y_test))
        all_set[lang] = _X_test,_y_test
    return all_set


class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = tokenizer.model_max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]

        tokenized_inputs = self.tokenizer(text, is_split_into_words=True, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=True)

        input_ids = tokenized_inputs['input_ids'].squeeze()
        attention_mask = tokenized_inputs['attention_mask'].squeeze()

        labels = []
        word_ids = tokenized_inputs.word_ids(batch_index=0)  # Get word_ids for the current sequence
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:  # Special tokens
                labels.append(-100)  # PyTorch's convention to ignore these labels during loss calculation
            elif word_idx != previous_word_idx:  # New word
                labels.append(self.label2id[word_labels[word_idx]])
            else:  # Subtoken of the previous word
                labels.append(self.label2id[word_labels[word_idx]])
            previous_word_idx = word_idx

        label_tensor = torch.LongTensor(labels)

        # Ensure label_tensor is padded/truncated to the correct length
        # This might not be necessary if your tokenizer already ensures correct padding
        # but is here for safety
        # padded_label_tensor = torch.full((self.max_length,), fill_value=-100, dtype=torch.long)
        # padded_label_tensor[:len(labels)] = label_tensor[:self.max_length]

        return input_ids, attention_mask, label_tensor

def collate_fn(batch, pad_token_id):
    input_ids, attention_masks, labels = zip(*batch)

    # Pad sequences so they match the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Assuming labels are already tensors. If not, you might need to convert or pad them here
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # Use -100 for ignored index in CrossEntropyLoss

    return input_ids_padded.to(device), attention_masks_padded.to(device), labels_padded.to(device)
