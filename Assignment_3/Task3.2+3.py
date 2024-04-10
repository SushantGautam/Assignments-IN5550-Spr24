from utils_summary import compute_metrics, json_folder_to_df_prompt_experiments
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import os
import json
import itertools

# Define argparse to accept checkpoint argument from CLI
parser = argparse.ArgumentParser(
    description="Train the model with a specified checkpoint.")
parser.add_argument("--checkpoint", type=str, choices=[
                    "facebook/opt-6.7b", "mistralai/Mistral-7B-v0.1", "mistralai/Mistral-7B-Instruct-v0.2", "google/flan-t5-xxl"], help="HF/checkpoint to use for the model")
parser.add_argument("--multi_shot", action='store_true',
                    help="Whether to use multiple prompts for each article")
args = parser.parse_args()

if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")

checkpoint = args.checkpoint
print(f"Using checkpoint: {checkpoint}")

# Load model and tokenizer based on the specified checkpoint
if checkpoint == "google/flan-t5-xxl":
    model= AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if checkpoint == "facebook/opt-6.7b" :
    folder_name = "data_tmp/opt_zero_shot"
elif checkpoint == "mistralai/Mistral-7B-v0.1":
    folder_name = "data_tmp/mistral_v01"
elif checkpoint == "mistralai/Mistral-7B-Instruct-v0.2":
    folder_name = "data_tmp/mistral_v02"
elif checkpoint == "google/flan-t5-xxl":
    folder_name = "data_tmp/flan_t5"

os.makedirs(folder_name, exist_ok=True)

print("Each row will be saved as a separate JSON file in the folder: ", folder_name)


df = pd.read_csv("data/train_test_split_1k.csv")

if checkpoint == "mistralai/Mistral-7B-Instruct-v0.2" or checkpoint == "google/flan-t5-xxl": 
    prompts = [ # instruction model
        "Summarize the following article: ",
        "Summarize the main points of the following article, focusing on the key events, findings, or arguments presented: ",
        "Provide a concise summary of this article, highlighting the most important information and conclusions in maximum 200 words: ",
        "Analyze the content of the article, providing a detailed summary that includes the background of the topic, key arguments or evidence presented, and the implications of the findings. Highlight any recommendations or future directions mentioned: ",
        "In maximum 200 words, transform the article into a comprehensive summary, addressing the who, what, where, when, and why. Discuss the significance of the research or events, the main arguments or theories presented, and any counterpoints or criticisms: ",
        ]
else:
    prompts = [
        'The plain-text summary sentence without URLs or lists is : ```',
        'The main argument/summary in plain text, avoiding URLs and lists is : ```',
        'The summary in plain text, up to 200 words, without URLs or lists is : ```',
        'The detailed summary in plain text, including background and implications, without URLs or lists is : ```',
        ' 200 words, the summary in plain text the who, what, where, when, why, avoiding URLs or lists is : ```',
    ]


few_shot_example = """
Given the following example of an article and its summary.
Article: ```The global economy is facing unprecedented challenges due to the combined impact of the COVID-19 pandemic, geopolitical tensions, and climate change. Economic growth has slowed considerably across both developed and developing nations. Supply chain disruptions and inflationary pressures are exacerbating economic inequalities, making it harder for low-income families to afford basic necessities. Governments worldwide are responding with a mix of fiscal stimuli, policy reforms, and support packages aimed at stabilizing markets and fostering sustainable growth.```
Summary: ```The global economy is struggling due to COVID-19, geopolitical tensions, and climate change, leading to slower growth, supply chain issues, and increased inequality. Governments are countering these challenges with various support measures.```"""



tokenizer.pad_token = tokenizer.eos_token
batch_size = 15   # should be multiple of prompt_length (5) to prevent fragmented json saving
batch_prompts = []
batch_meta_info = []  # To keep track of row_id for each prompt in the batch

for row_id, row in tqdm(df.iterrows(), total=df.article.count(), desc="Generating text"):
    json_file = f"{folder_name}/{row_id}.json"
    if os.path.exists(json_file):
        continue

    for idx, prompt in enumerate(prompts):
        final_prompt = f"Article: {row.article[:2000]}.\n\n  {prompt}"
        if args.multi_shot:
            final_prompt = f"{few_shot_example} \n\n Article: {row.article[:2000]}.\n\n  {prompt}"
        if checkpoint == "mistralai/Mistral-7B-Instruct-v0.2":
            final_prompt = f"<s>[INST]{prompt}```{row.article}```[/INST]"
        if checkpoint == "google/flan-t5-xxl":
            final_prompt = f"{prompt} {row.article[:2200]}\n Summary: \n"
        # we truncate the input to 512tokens, 2400 chars, 400 words
        # final_prompt = f"{prompt} {row.article[:2200]}\n Summary: \n"
        batch_prompts.append(final_prompt)
        # Store row_id and prompt_id for each output to be generated
        batch_meta_info.append((row_id, idx))

        if len(batch_prompts) >= batch_size or (row_id == df.index[-1] and idx == len(prompts) - 1):
            # Process the batch
            if checkpoint == "mistralai/Mistral-7B-Instruct-v0.2":
                model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for i, (row_id, prompt_id) in enumerate(batch_meta_info):
                    output = decoded_outputs[i].split("[/INST]")[-1].split("```")[-1]
                    outputs.append(
                        {"row_id": row_id, "prompt_id": prompt_id, "text": output})
            elif checkpoint == "google/flan-t5-xxl":
                model_inputs = tokenizer([final_prompt], return_tensors="pt").to("cuda")
                generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                for i, (row_id, prompt_id) in enumerate(batch_meta_info):
                    outputs.append(
                        {"row_id": row_id, "prompt_id": prompt_id, "text": output})
            else:
                model_inputs = tokenizer(batch_prompts, return_tensors="pt",
                                        padding=True, truncation=True, max_length=1024).to("cuda")
                generated_ids = model.generate(
                    **model_inputs, max_new_tokens=256, do_sample=True, pad_token_id=tokenizer.eos_token_id)
                decoded_outputs = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)

                outputs = []
                for i, (row_id, prompt_id) in enumerate(batch_meta_info):
                    output = decoded_outputs[i].split(": ```")[-1]
                    outputs.append(
                        {"row_id": row_id, "prompt_id": prompt_id, "text": output})

            # Group by row_id and save to JSON
            for row_id, group in itertools.groupby(outputs, lambda x: x["row_id"]):
                group_list = list(group)
                data = {"data": [[g["row_id"], g["prompt_id"], g["text"]]
                                 for g in group_list]}
                with open(f"folder_name/{row_id}.json", 'w') as f:
                    json.dump(data, f)

            batch_prompts = []
            batch_meta_info = []


gen_summaries = json_folder_to_df_prompt_experiments(folder_name)

for prompt, group_df in gen_summaries.groupby('prompt#'):
    print(f"Prompt#: {prompt}")
    scores = compute_metrics(group_df.summary.to_list(), [
                             df.loc[e].summary for e in group_df.org_row_ID])
    print(scores)