from tqdm import tqdm 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
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
parser.add_argument("--checkpoint", type=str, help="HF/checkpoint to use for the model, like SushantGautam/roberta-large_synt_flan")
args = parser.parse_args()
if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")

checkpoint="google/flan-t5-xxl"
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

folder_name = "data_tmp/6.flan.generations"
os.makedirs(folder_name, exist_ok=True)


df= pd.read_csv("data/blind.csv")


prompt_ins= "Provide a concise summary of the text provided, highlighting the most important information and conclusions in maximum 100 words:"

tokenizer.pad_token = tokenizer.eos_token
gen_len= 5

batch_size = 4   # Adjust based on your GPU capacity
batch_prompts = []
batch_row_ids = []  # To keep track of row_id for each prompt in the batch

for row_id, row in tqdm(df[::-1].iterrows(), total=df.article.count(), desc="Generating text"):
    json_file = f"{folder_name}/{row_id}.json"
    if os.path.exists(json_file):
        continue
     
    final_prompt = f"{prompt_ins} {row.article}\n Summary: \n" # FLAN
    batch_prompts.append(final_prompt)  # Prepare to generate 5 outputs for this prompt
    batch_row_ids.append(row_id)  # Store row_id for each output to be generated

    if len(batch_prompts) >= batch_size:
        # Tokenize and process the batch when it reaches the batch size
        model_inputs = tokenizer(batch_prompts, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512, padding=True).to("cuda")
        generated_ids = model.generate(**model_inputs, max_new_tokens=1024, num_return_sequences=gen_len, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, pad_token_id=tokenizer.eos_token_id)
        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        row_buck= [num for num in batch_row_ids for _ in range(gen_len)]
        outputs = [(row_buck[i], decoded_outputs[i]) for i in range(len(decoded_outputs))]
        for row_id, group in itertools.groupby(outputs, lambda x: x[0]):
            group_list = list(group)
            data = {"data": [g[1] for g in group_list]}
            with open(f"{folder_name}/{row_id}.json", 'w') as f:
                json.dump(data, f)
        batch_prompts = []
        batch_row_ids = []


import glob, json
all_jsons = glob.glob(f"{folder_name}/"+"**.json")
json_data_list = []

for json_file in all_jsons:
    with open(json_file, 'r') as f:
        row_id = int(json_file.split("/")[-1].split(".")[0])
        article = df.loc[row_id].article
        json_content = json.load(f)
        summaries= json_content['data']
        for i in range(len(summaries)):
            json_data_list.append([row_id, i, article,summaries[i].strip()])

dfx = pd.DataFrame(json_data_list, columns=["org_row_ID", "candidate#", "article", "summary"])
# dfx.to_csv("data_tmp/6.FLAN_blind-candidate.csv", index=False)

checkpoint = "logs/FacebookAI/roberta-large_synt_flan_/checkpoint-6306/"


i_max_length  = 512
budget = i_max_length 

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
dfx[checkpoint] = predictions
# dfx.to_csv("data_tmp/6.FLAN_blind-candidate_scored.csv", index=False)


#8]:


best_summaries = dfx.loc[dfx.groupby('org_row_ID')['roberta-large_synt_flan_'].idxmax()]
best_summaries = best_summaries[['article', 'summary']].reset_index(drop=True)
best_summaries
best_summaries.to_csv("final_submission.csv", index=False)


#18]:


import gzip 
with gzip.open( 'final_submission.jsonl.gz', 'wt', encoding='utf-8') as f:
    f.write(best_summaries.to_json(orient='records', lines=True, default_handler=str, force_ascii=False))
