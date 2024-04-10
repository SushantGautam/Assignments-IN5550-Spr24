#!/usr/bin/env python
# coding: utf-8

#3]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.utils import shuffle
from datasets import Dataset as datasetx
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import  AutoTokenizer, AutoModelForSequenceClassification,EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import evaluate
import numpy as np
import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(
    description="Train the model with a specified checkpoint.")
parser.add_argument("--checkpoint", type=str, choices=[
                    "google-bert/bert-large-cased", "FacebookAI/roberta-large"], help="HF/checkpoint to use for the model")
parser.add_argument("--generation_csv", type=str, help="CSV file with the generations from previous task, hint: saved in data_tmp/5.3synthtic_scored.csv")
parser.add_argument("--keep_reference_summary", action='store_true',
                    help="Whether to use use reference_summary as well")
args = parser.parse_args()

if args.generation_csv is None:
    raise ValueError(
        "Please provide a valid CSV path using --generation_csv argument.")
if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")

keep_reference_summary= args.keep_reference_summary
checkpoint = args.checkpoint

run_name=  checkpoint+'_synt_flan'+ "+_with_reference_summ" if keep_reference_summary else  "logs/"+ checkpoint+'_synt_flan'
print("Los and model will be saved at: logs/"+ run_name)

synthtic_df = pd.read_csv(args.generation_csv )
list1 = synthtic_df.rougeL
min_value, max_value = min(list1), max(list1)
synthtic_df['score'] =[(x - min_value) / (max_value - min_value) * (6) + 1 for x in list1]


#25]:


synthtic_df['score'].describe()


#26]:


df_= pd.read_csv("data/feedback.csv").drop_duplicates().dropna()
df_['word_count'] = df_['summary'].apply(lambda x: len(x.split()))
#filtering some edge cases; no much motive want to reduce numbers for training ;)
df_ = df_[(df_['word_count'] > 5) & (df_['word_count'] <= 200)] # greater than 5
df_['article_word_count'] = df_['article'].apply(lambda x: len(x.split()))
df_ = df_[(df_['article_word_count'] > 40)]  # some articles are too long anyway
#overall stat min	1.000000, 25%	4.000000, 50%	5.000000, 75%	6.000000, max	7.000000
df_.describe()


#27]:


df_['score'] = df_.overall


#28]:


if keep_reference_summary:
    ref_df = synthtic_df[['article', 'reference']].drop_duplicates()
    ref_df.rename({'reference': 'summary'}, axis=1, inplace=True)
    ref_df['score']= 1 #correctness score
    df = pd.concat([synthtic_df[['article', 'summary', 'score']], ref_df, df_[['article', 'summary', 'score']]])
else:
    df = pd.concat([synthtic_df[['article', 'summary', 'score']], df_[['article', 'summary', 'score']]])
    
df= df.dropna()
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['score'].astype(int))


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
calculate_dynamic_score = lambda examples, slist: [
    (sum(scores) - len(scores)) / (len(slist) * 6) for scores in zip(*[examples[score_type] for score_type in slist])
]
i_max_length  = 512
budget = i_max_length  -1

def process_pair(examples):
    processed_examples = []
    for article, summary in zip(examples['article'], examples['summary']):
        article_tokens, summary_tokens = map(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True), [article, summary])
        if (total_length := len(summary_tokens) + len(article_tokens)-1) > budget:
            summary_budget = int(min(len(summary_tokens), budget/2)) # 
            article_tokens = article_tokens[:budget - summary_budget] 
            if article_tokens[-1]!=tokenizer.sep_token_id:  # check of SEP is lost by truncation
                    article_tokens +=[tokenizer.sep_token_id]
            summary_tokens = summary_tokens[:summary_budget]
            if summary_tokens[-1]!=tokenizer.sep_token_id: # check of SEP is lost by truncation
                    summary_tokens +=[tokenizer.sep_token_id]
        processed_examples.append(article_tokens + summary_tokens[1:] ) # ignore starting token of article_tokens
    return {'input_ids': processed_examples, 'labels': calculate_dynamic_score(examples, slist=['score'])} # attenttion is automatically calculated
    
df_train = datasetx.from_pandas(train_df).map(process_pair, batched=True)
df_val = datasetx.from_pandas(val_df).map(process_pair, batched=True)


#29]:


max([len(e['input_ids']) for e in df_train ])


#30]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return { "RMSE": rmse, "MAE": mae, "R2": r2
    }
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#31]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1) #regression

training_args = TrainingArguments(
    output_dir= "logs/"+ run_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy ="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    fp16= True,
    metric_for_best_model = 'loss',   
    load_best_model_at_end = True,
    run_name= checkpoint+'_synt_flan'+ "_with_reference_summ" if keep_reference_summary else checkpoint+'_synt_flan',
    auto_find_batch_size= True,

) 

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=df_train,
    eval_dataset=df_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(3, 0.0)]
)

trainer.train()
print("Model saved at: logs/"+ run_name)
