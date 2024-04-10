from tqdm import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import itertools
import os, json
from datasets import Dataset as datasetx

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer as Predictor
from transformers import  AutoTokenizer, AutoModelForSequenceClassification,EarlyStoppingCallback
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse

parser = argparse.ArgumentParser(description="Evaluate generated summaries using trained scoring models.")
parser.add_argument("--generation_csv", type=str, help="CSV file with the generations from previous task, hint: saved in data_tmp/4.3generations.csv")
parser.add_argument("--models", type=str, nargs='*', help="HF/local checkpoints of the models to use for scoring, like SushantGautam/roberta-large_accuracy-coverage, Separate with space")

args = parser.parse_args()


if args.generation_csv is None:
    raise ValueError(
        "Please provide a valid CSV path using --generation_csv argument.")

if args.models is None:
    raise ValueError(
        "Please provide a valid models path using --models argument. Separate with space if multiple models are provided.")

df = pd.read_csv("data/train_test_split_1k.csv")

dfx= pd.read_csv(args.generation_csv).dropna() # same candidates for synthetic as well in task 4
dfx = dfx[dfx['org_row_ID'].isin(df.index)]


checkpoints = args.models


i_max_length  = 512
budget = i_max_length 
for checkpoint in checkpoints:
    print("Scoring with: ", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1) #regression
    
    def process_pair(examples):
        processed_examples = []
        for article, summary in zip(examples['article'], examples['summary']):
            article_tokens, summary_tokens = map(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True), [article, summary])
            if (total_length := len(summary_tokens) + len(article_tokens)-1) > budget:
                summary_budget = min(len(summary_tokens), budget/2) # 
                article_tokens = article_tokens[:budget - summary_budget] 
                if article_tokens[-1]!=tokenizer.sep_token_id:  # check of SEP is lost by truncation
                        article_tokens +=[tokenizer.sep_token_id]
                summary_tokens = summary_tokens[:summary_budget]
                if summary_tokens[-1]!=tokenizer.sep_token_id: # check of SEP is lost by truncation
                        summary_tokens +=[tokenizer.sep_token_id]
            processed_examples.append(article_tokens + summary_tokens[1:] ) # ignore starting token of article_tokens
        return {'input_ids': processed_examples} # attenttion is automatically calculated
    
    
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['article'], examples['summary'], padding="max_length", truncation=True, max_length=512,)
        return tokenized_inputs
        
    df_eval = datasetx.from_pandas(dfx).map(preprocess_function, batched=True)
    predictions, _, _ = Predictor(model=model).predict(test_dataset=df_eval)
    dfx[checkpoint.split('/')[-3]] = predictions
dfx.to_csv("data_tmp/4.3candiates_scored.csv", index=False)


#7]:


dfx = pd.read_csv("data_tmp/4.3candiates_scored.csv")
dfx.describe()


from utils_summary import compute_metrics
for column in dfx.columns[-4:]:
    # For each column, compute best summary
    dfx_scores = dfx.copy()
    dfx_scores['selected_score'] = dfx_scores[column]
    best_summaries = dfx_scores.loc[dfx_scores.groupby('org_row_ID')['selected_score'].idxmax()]
    best_summaries = best_summaries[['org_row_ID', 'article', 'summary']]    
    scores = compute_metrics(best_summaries.summary.to_list(), [df.loc[e].summary for e in best_summaries.org_row_ID])
    print(f"Scores for {column}:")
    print(scores)
