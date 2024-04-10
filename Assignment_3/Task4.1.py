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
parser.add_argument("--correctness_metric", type=str, nargs=2, choices=[
                    'overall', 'coherence', 'accuracy', 'coverage'], help="Correctness metrics to evaluate. Out of 'overall', 'accuracy', 'coverage', 'coherence'")
args = parser.parse_args()
if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")
if args.correctness_metric is None:
    raise ValueError(
        "Please provide a valid correctness metrics using --correctness_metric argument.")


df_= pd.read_csv("data/feedback.csv").drop_duplicates().dropna()
df_['word_count'] = df_['summary'].apply(lambda x: len(x.split()))

#filtering some edge cases; no much motive want to reduce numbers for training ;)
df_ = df_[(df_['word_count'] > 5) & (df_['word_count'] <= 200)] # greater than 5
df_['article_word_count'] = df_['article'].apply(lambda x: len(x.split()))
df_ = df_[(df_['article_word_count'] > 40)]  # some articles are too long anyway

df= df_

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify= pd.cut(df['word_count'], bins=4, labels=False))

checkpoint = args.checkpoint # "google-bert/bert-large-cased" #FacebookAI/roberta-large, google-bert/bert-large-cased
# score_metrics=['overall', 'coherence'] #'overall', 'accuracy', 'coverage', 'coherence'
score_metrics= args.correctness_metric
print(f"Using checkpoint: {checkpoint} and correctness metrics: {score_metrics}")

output_dir = "logs/"+ checkpoint+'_'+"-".join(score_metrics)
print("Training logs and model will be saved in: ", output_dir)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
calculate_dynamic_score = lambda examples, slist: [
    (sum(scores) - len(scores)) / (len(slist) * 6) for scores in zip(*[examples[score_type] for score_type in slist])
]
i_max_length  = 512
budget = i_max_length

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
    return {'input_ids': processed_examples, 'labels': calculate_dynamic_score(examples, slist=score_metrics)} # attenttion is automatically calculated

    
df_train = datasetx.from_pandas(train_df).map(process_pair, batched=True)
df_val = datasetx.from_pandas(val_df).map(process_pair, batched=True)


# 3]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return { "RMSE": rmse, "MAE": mae, "R2": r2
    }
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# 4]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1) #regression

training_args = TrainingArguments(
    output_dir= output_dir,
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
    run_name=checkpoint+'_'+"-".join(score_metrics),
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


print("Logs and model are saved in: ", output_dir)