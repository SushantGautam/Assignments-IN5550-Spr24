import pandas as pd
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, pipeline,  AutoModelForTokenClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from seqeval.scheme import IOB2
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser

device = "cuda" if torch.cuda.is_available() else "cpu"
# device
id2label = {0: 'O',1: 'I-ORG', 2: 'I-PER', 3: 'I-LOC', 4: 'B-ORG', 5: 'B-PER', 6: 'B-LOC'}
label2id = {v:i for i,v in id2label.items()}

def pad_batch(batch, pad_token_id):
    max_length = max([len(sentence) for sentence in batch])
    padded_batch = [sentence + [pad_token_id] * (max_length - len(sentence)) for sentence in batch]
    return torch.tensor(padded_batch, dtype=torch.long)


def dump_to_tsv(pred_tags_mapped_all, _data):
    result = ""
    # Iterate through each list of tags and corresponding words
    for pred_tags, data_tags in zip(pred_tags_mapped_all, _data):
        # Join the tags and words together with newline separator
        tagged_words = "\n".join([f"{word}\t{tag}" for word, tag in zip(data_tags, pred_tags)])
        # Append the tagged words to the result string with double newline separator
        result += tagged_words + "\n\n"
    return result




def predict(model_dir, data_file, batch_size = 1):
    data = open(data_file, 'r', encoding='utf-8').read() ############
    _data = [[line for line in block.split('\n') if line.strip()] for block in data.split('\n\n')]

    # label2id
    
    model_path=model_dir 
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=128)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), ignore_mismatched_sizes=True).to(device)


    model.eval()
    total_loss = 0
    predictions = []
    word_ids = []
        
    with torch.no_grad():
        for i in range(0, len(_data), batch_size):  # Assuming batch size of 16
            batch = _data[i:i+batch_size]
            tokenized_inputs = tokenizer(batch, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokenized_inputs['input_ids'].to(device)
            attention_mask = tokenized_inputs['attention_mask'].to(device)
    
            # Since we don't have labels, we'll only predict
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = logits.argmax(dim=2).cpu().numpy()
    
            # Process predictions
            for j, mask in enumerate(attention_mask):
                # Remove predictions for padding tokens and special tokens
                true_prediction = [pred for pred, m in zip(prediction[j], mask) if m == 1]
                cropped_predictions = true_prediction[1:-1]  # Remove [CLS] and [SEP] token predictions
                predictions.append(cropped_predictions)
                word_ids.append(tokenized_inputs.word_ids(j)[1:-1])
    
            # if(i%100 == 0):
            #     print("Now in: ",i)
    
    # Convert predictions to tag sequences
    pred_tags = [[id2label[tag_id] for tag_id in seq] for seq in predictions]
    print(len(_data), len(pred_tags), len(word_ids))
    
    pred_tags_mapped_all =[]
    for data_idx, data_val in enumerate(_data):
        idx= data_idx
        pred_tags_mapped = []
        word_ids__ = word_ids[idx]
    
        last_word_idx = None
    
        for idx_, val in enumerate(word_ids__):
            # print(idx_, val)
            if val is None:
                pass
            else:
                if val != last_word_idx:
                    pred_tags_mapped.append(pred_tags[idx][idx_])
                    # print("adding ", idx_, val, pred_tags[idx][idx_])
                last_word_idx = val
                # print("setting last word idx", idx_)
    
        # print(pred_tags_mapped)
        pred_tags_mapped_all.append(pred_tags_mapped)

    # Dump data to TSV format
    tsv_data = dump_to_tsv(pred_tags_mapped_all, _data)
    with open("pred_tags_mapped.tsv", "w") as file:
            file.write(tsv_data)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        required=False,
        help="path to a folder with the model.",
        default='/cluster/work/projects/ec30/fernavr/best_model',
    )

    parser.add_argument(
        "--data",
        "-d",
        help="path to a .tsv file with one column.",
        required=False,
        default='/fp/projects01/ec30/IN5550/obligatories/2/surprise/surprise_test_set.tsv'
    )
    args = parser.parse_args()
    predict(model_dir=args.model, data_file=args.data)