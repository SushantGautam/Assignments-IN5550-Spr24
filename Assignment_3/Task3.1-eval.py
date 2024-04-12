
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.utils import shuffle
from datasets import Dataset as datasetx
import os
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq,  DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelWithLMHead
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
import evaluate
import numpy as np
import argparse
from tqdm import tqdm 
from utils_summary import compute_metrics

# ]:
# Define argparse to accept checkpoint argument from CLI
parser = argparse.ArgumentParser(
    description="Train the model with a specified checkpoint.")
parser.add_argument("--checkpoint", type=str, choices=[
                    "SushantGautam/t5-base", "SushantGautam/opt-350m-lora"], help="HF/checkpoint to use for the model")
args = parser.parse_args()

if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")

print("Calculating ROUGE and BERTScore for the generated summaries with model: ", args.checkpoint)
        
if args.checkpoint == "SushantGautam/t5-base":
    prefix = "summarize: "
    model_path = "SushantGautam/t5-base"
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    i_max_length, o_max_length = 512, 256
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to("cuda")
else:
    from peft import AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained("SushantGautam/opt-350m-lora").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model.eval()
    i_max_length, o_max_length = 350, 350


df_test = pd.read_csv("data/train_test_split_1k.csv")

prediction, reference = [], []

for row_id, row in tqdm(df_test.iterrows(), total=df_test.article.count(), desc="Generating text"):
    if args.checkpoint == "SushantGautam/t5-base":
        separator = tokenizer.eos_token
    else:
        separator = "\nTL;DR\n"  ######## check logic around this
    if args.checkpoint == "SushantGautam/t5-base":
        prompt = f"{prefix}{row.article}"
    else:
        prompt = f"{row.article[:2000]}{separator}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    output = model.generate(**inputs, max_new_tokens=256, num_beams=5, early_stopping=True) ######## check logic around this
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    actual_output = raw_output
    if not args.checkpoint == "SushantGautam/t5-base": # for OPT model
        actual_output = raw_output.split(separator)[-1]
    prediction.append(actual_output)
    reference.append(row.summary)

results = compute_metrics(predictions=prediction, references=reference)
print("Scores: ", results)

