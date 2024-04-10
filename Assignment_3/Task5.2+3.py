
from tqdm import tqdm 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import itertools
import os, json
import argparse

parser = argparse.ArgumentParser(description="Evaluate generated summaries using trained scoring models.")
parser.add_argument("--checkpoint", type=str, choices=[
                    "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"], help="HF/checkpoint to use for the model")
args = parser.parse_args()
if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")


checkpoint=args.checkpoint
if checkpoint == "google/flan-t5-xxl":
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)



df = pd.read_csv("data/train_test_split_1k.csv")
folder_name = "data_tmp/5.2generations"

prompt_ins= "Provide a concise summary of the text provided, highlighting the most important information and conclusions in maximum 100 words:"

tokenizer.pad_token = tokenizer.eos_token
gen_len= 5

batch_size = 5   # Adjust based on your GPU capacity
batch_prompts = []
batch_row_ids = []  # To keep track of row_id for each prompt in the batch

for row_id, row in tqdm(df.iterrows(), total=df.article.count(), desc="Generating text"):
    json_file = f"{folder_name}/{row_id}.json"
    if os.path.exists(json_file):
        continue
    
    if checkpoint == "google/flan-t5-xxl":
        final_prompt =  f"{prompt_ins} {row.article}\n Summary: \n" # FLAN
    else:
        final_prompt =  f"<s>[INST]{prompt_ins}``` {row.article}```[/INST]" , # mistral
    batch_prompts.append(final_prompt)  # Prepare to generate 5 outputs for this prompt
    batch_row_ids.append(row_id)  # Store row_id for each output to be generated

    if len(batch_prompts) >= batch_size:
        if checkpoint == "google/flan-t5-xxl":
            model_inputs = tokenizer(batch_prompts, add_special_tokens=True, return_tensors="pt", truncation=True, max_length=512, padding=True).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=1024, num_return_sequences=gen_len, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, pad_token_id=tokenizer.eos_token_id)
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            row_buck= [num for num in batch_row_ids for _ in range(gen_len)]
            outputs = [(row_buck[i], decoded_outputs[i]) for i in range(len(decoded_outputs))]
        else: #mistral
            model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
            generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            row_buck= [num for num in batch_row_ids for _ in range(gen_len)]
            outputs = [(row_buck[i], decoded_outputs[i].split("[/INST]")[-1].split("```")[-1]) for i in range(len(decoded_outputs))]

        for row_id, group in itertools.groupby(outputs, lambda x: x[0]):
            group_list = list(group)
            data = {"data": [g[1] for g in group_list]}
            with open(f"{folder_name}/{row_id}.json", 'w') as f:
                json.dump(data, f)
        batch_prompts = []
        batch_row_ids = []



import glob, json
import evaluate
import numpy as np

all_jsons = glob.glob(f"{folder_name}"+"/**.json")
json_data_list = []

for json_file in all_jsons:
    with open(json_file, 'r') as f:
        row_id = int(json_file.split("/")[-1].split(".")[0])
        try:
            json_content = json.load(f)
        except:
            continue
        summaries= json_content['data']
        for i in summaries:
            json_data_list.append([row_id, df.loc[row_id].article, df.loc[row_id].summary, i.strip()])



predictions, references=  [e[3] for e in json_data_list],  [e[2] for e in json_data_list]



rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, use_aggregator=False, references=references, use_stemmer=True)


rougeL= rouge_result['rougeL']
final_data = [old_data + [rougeL[i]] for i, old_data in enumerate(json_data_list)]


final_data = pd.DataFrame(final_data, columns=["org_row_ID","article", "reference", "summary", "rougeL"])
final_data.to_csv("data_tmp/5.3synthtic_scored.csv", index=False)

print("Generated summaries are saved in data_tmp/5.3synthtic_scored.csv")


# ]:




