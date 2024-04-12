
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

# ]:
# Define argparse to accept checkpoint argument from CLI
parser = argparse.ArgumentParser(
    description="Train the model with a specified checkpoint.")
parser.add_argument("--checkpoint", type=str, choices=[
                    "google-t5/t5-base", "ybelkada/opt-350m-lora"], help="HF/checkpoint to use for the model")
args = parser.parse_args()

if args.checkpoint is None:
    raise ValueError(
        "Please provide a valid checkpoint name using --checkpoint argument.")

print("Calculating ROUGE and BERTScore for the generated summaries with model: ", args.checkpoint)
        
if args.checkpoint == "google-t5/t5-base":
    prefix = "summarize: "
    model_path = "google-t5/t5-base"
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    i_max_length, o_max_length = 512, 256
else:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    import torch
    model = AutoPeftModelForCausalLM.from_pretrained("SushantGautam/tmp-opt-350m-lora").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    model.eval()
    i_max_length, o_max_length = 350, 350


df_test = pd.read_csv("data/train_test_split_1k.csv")

if args.checkpoint == "google-t5/t5-base":
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=i_max_length, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=o_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    df_train = datasetx.from_pandas(train_df).map(preprocess_function, batched=True)
    df_val = datasetx.from_pandas(val_df).map(preprocess_function, batched=True)

else:
    budget = i_max_length
    separator="\nTL;DR\n"
    separator_tokens = tokenizer.encode(separator, add_special_tokens=False)

    def preprocess_function(examples):
        processed_examples = []

        for article, summary in zip(examples['article'], examples['summary']):
            # Tokenize summary first to ensure it is not truncated
            summary_tokens = tokenizer.encode(summary, add_special_tokens=False)
            summary_budget = int(min(len(summary_tokens), budget/2))
            summary_tokens = summary_tokens[:summary_budget]
            # Calculate remaining budget for the article
            remaining_budget = budget - len(summary_tokens) - len(separator_tokens)
            
            # Tokenize the article within the remaining budget
            article_tokens = tokenizer.encode(article, add_special_tokens=False, truncation=True, max_length=remaining_budget)

            # Combine article tokens, separator tokens, and summary tokens
            tokens = article_tokens + separator_tokens + summary_tokens
            
            # Ensure the combined tokens do not exceed the budget, just in case
            tokens = tokens[:budget]
            processed_examples.append(tokens)
        return processed_examples


prediction, reference = [], []

for row_id, row in tqdm(df_test.iterrows(), total=df_test.article.count(), desc="Generating text"):
    if args.checkpoint == "google-t5/t5-base":
        separator = tokenizer.eos_token
    else:
        separator = "\nTL;DR\n"  ######## check logic around this
    if args.checkpoint == "google-t5/t5-base":
        prompt = f"{prefix}{row.article}"
    else:
        prompt = f"{row.article}{separator}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    output = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True) ######## check logic around this
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True)
    actual_output = raw_output[len(prompt):]
    prediction.append(actual_output)
    reference.append(row.summary)


rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Compute ROUGE scores
rouge_result = rouge.compute(predictions=prediction, references=reference, use_stemmer=True)
bertscore_result = bertscore.compute(predictions=prediction, references=reference, lang="en", rescale_with_baseline=True)

print("ROUGE Scores: ", rouge_result)
print("BERTScore: ", bertscore_result)



