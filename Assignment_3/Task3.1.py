#!/usr/bin/env python
# coding: utf-8

#1]:


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


df_= pd.read_csv("train.csv").drop_duplicates().dropna()
df_['word_count'] = df_['summary'].apply(lambda x: len(x.split()))
df_ = df_[(df_['word_count'] > 5) & (df_['word_count'] <= 200)] # greater than 5
df_['article_word_count'] = df_['article'].apply(lambda x: len(x.split()))
df_ = df_[(df_['article_word_count'] > 40)]  # some articles are too long anyway

df_test= df_.sample(n=1, random_state=42).head(5000) # exclude 5K 

df = df_.drop(index=df_test.index)
df


#3]:


train_df, val_df = train_test_split(df, test_size=0.2, stratify= pd.cut(df['word_count'], bins=1, labels=False))


if args.checkpoint == "google-t5/t5-base":
    prefix = "summarize: "
    model_path = "google-t5/t5-base"
    checkpoint = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    i_max_length, o_max_length = 512, 256
else:
    checkpoint= args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    from peft import AutoPeftModelForCausalLM

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM , inference_mode=False, r=4, lora_alpha=16, lora_dropout=0.1
    )
    model = get_peft_model(AutoPeftModelForCausalLM.from_pretrained(checkpoint).to("cuda"), peft_config)
    val_df= df_test # validate only on test data as it is not real validation anyway
    i_max_length, o_max_length = 350, 350
    




if args.checkpoint == "google-t5/t5-base":
    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=i_max_length, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=o_max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    df_train = datasetx.from_pandas(train_df).map(preprocess_function, batched=True)
    df_val = datasetx.from_pandas(val_df).map(preprocess_function, batched=True)

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute ROUGE scores
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Compute BERTScore
        bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", rescale_with_baseline=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        metrics_result = {"gen_len": np.mean(prediction_lens)}

        # Update the result dictionary with ROUGE and BERTScore results
        for key, value in rouge_result.items():
            metrics_result[f"eval_{key}"] = round(value, 4)
        for key, value in bertscore_result.items():
            if key in ["precision", "recall", "f1"]:
                metrics_result[f"eval_bert-score-{key}"] = np.mean(value).item()

        return metrics_result

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
        return {'input_ids': processed_examples, 'labels':processed_examples} # attention is automatically calculated

    df_train = datasetx.from_pandas(train_df).map(preprocess_function, batched=True)
    df_val = datasetx.from_pandas(val_df).map(preprocess_function, batched=True)

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    # from bert_score import BERTScorer
    # scorer_bert = BERTScorer(lang="en", rescale_with_baseline=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        # predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = [[int(token) if token != -100 else tokenizer.pad_token_id for token in pred] for pred in predictions]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Compute BERTScore
        bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", rescale_with_baseline=True)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        metrics_result = {"gen_len": np.mean(prediction_lens)}

        # Update the result dictionary with ROUGE and BERTScore results
        for key, value in rouge_result.items():
            metrics_result[f"eval_{key}"] = round(value, 4)
        for key, value in bertscore_result.items():
            if key in ["precision", "recall", "f1"]:
                metrics_result[f"eval_bert-score-{key}"] = np.mean(value).item()

        return metrics_result


    

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)


print("Saving logs in: ", "logs/"+ args.checkpoint)

if args.checkpoint == "google-t5/t5-base":
    training_args = Seq2SeqTrainingArguments(
        output_dir= "logs_T5/"+ checkpoint,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        logging_steps= 50,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=40,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        metric_for_best_model = 'rougeL',   
        greater_is_better = True, 
        load_best_model_at_end = True,
        generation_max_length = o_max_length
    ) 

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=df_train,
        eval_dataset=df_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(3, 0.0)]
    )

else:
    training_args = TrainingArguments(
        output_dir= "logs/"+ checkpoint,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        logging_steps= 50,
        per_device_train_batch_size=30,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        # predict_with_generate=True,
        metric_for_best_model = 'rougeL',   
        greater_is_better = True, 
        load_best_model_at_end = True,
        fp16=True,
        # generation_max_length = o_max_length
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

print("Logs and model are saved in: ", "logs/"+ args.checkpoint)