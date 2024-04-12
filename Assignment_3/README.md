# Assignment 3
## Fox Modules used:
```
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8 
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8 
```

## Training and evaluation
### Task 3.1
To train models in 3.1:
```python
Task3.1.py [-h] --checkpoint {google-t5/t5-base, ybelkada/opt-350m-lora}

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model. One of google-t5/t5-base, ybelkada/opt-350m-lora.
```

Two of the trained models reported in report are uploaded in HuggingFace model repository with ID:  SushantGautam/t5-base, SushantGautam/opt-350m-lora

To evaluate the models trained above on test set:

```python
Task3.1-eval.py [-h] --checkpoint {SushantGautam/t5-base, SushantGautam/opt-350m-lora} 

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model. One of SushantGautam/t5-base, SushantGautam/opt-350m-lora
```


### Task 3.2 and 3.3
To generate results in 3.2 and 3.3 using **facebook/opt-6.7b**,  **mistralai/Mistral-7B-v0.1**, **mistralai/Mistral-7B-Instruct-v0.2** or **google/flan-t5-xxl** with zero and few shot prompting.

```python
Task3.2+3.py [-h] --checkpoint {facebook/opt-6.7b,mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.2,google/flan-t5-xxl} [--multi_shot]

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model. One of facebook/opt-6.7b or mistralai/Mistral-7B-v0.1.
--multi_shot: Enables the use of multiple prompts for each article foir task 3.2.2. This option does not require a value. Dont include for zero shot prompting for 3.2.1. Doesn't apply when using mistralai/Mistral-7B-Instruct-v0.2 and google/flan-t5-xxl which only is implemneted for zero-shot for 3.3.
```
 The script creates a folder with a JSON for each data row. Also, those generation results from 3.2 and 3.3 are compiled to CSVs in data_tmp folder for comprehension.

### Task 4.1 and 4.2
To generate results in 4.1 and 4.2 using google-bert/bert-large-cased or FacebookAI/roberta-large.

```python
Task4.1+2.py [-h] --checkpoint {google-bert/bert-large-cased,FacebookAI/roberta-large} --correctness_metric {overall,coherence,accuracy,coverage}

**arguments:**
--correctness_metric: Metric(s) to be used for correctness score for task 4.1. Separated by space. 
--checkpoint: Specifies HF checkpoint to use for the model for task 4.2. One of google-bert/bert-large-cased or FacebookAI/roberta-large.
**example:**
python Task4.1+2.py --checkpoint FacebookAI/roberta-large  --correctness_metric accuracy coherence
```
Four of the trained models reported in report are uploaded in HuggingFace model repository with ID: SushantGautam/roberta-large_accuracy-coverage, SushantGautam/roberta-large_overall-coherence, SushantGautam/bert-large-cased_accuracy-coverage, SushantGautam/bert-large-cased_overall-coherence

 The training logs are published in Weights and Biases at https://wandb.ai/sushantgautam/nlp_assignment

### Task 4.3a Generate 5 candidate summaries for test set 
To generate results in 4.3 using google/flan-t5-xxl or mistralai/Mistral-7B-Instruct-v0.2 with zero shot prompting.

```python
Task4.3.py [-h] --checkpoint {google/flan-t5-xxl,mistralai/Mistral-7B-Instruct-v0.2} --correctness_metric {overall,coherence,accuracy,coverage}

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model for task 4.2. One of "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"
**example:**
python Task4.3.py --checkpoint google/flan-t5-xxl
```
The generation results from the FLAN model reported in report is saved in "data_tmp/4.3generations.csv".

### Task 4.3b Evaluation with Scorer Model

```python
Task4.3-eval.py [-h] --generation_csv PATH  --models ...

**arguments:**
--generation_csv: CSV file with the generations from previous task, hint: saved in data_tmp/4.3generations.csv, 
--models: HF/local checkpoints of the models to use for scoring, like SushantGautam/roberta-large_accuracy-coverage, Separate with space
**example:**
python 4.3-eval.py --generation_csv data_tmp/4.3generations.csv --models SushantGautam/roberta-large_accuracy-coverage SushantGautam/roberta-large_overall-coherence SushantGautam/bert-large-cased_accuracy-coverage SushantGautam/bert-large-cased_overall-coherence
```
The script prints the metrics but also logs the intermediate scores for each candidate at "data_tmp/4.3candiates_scored.csv". 


### Task 5.2 and 5.3
To generate results in 5.3 using google/flan-t5-xxl or mistralai/Mistral-7B-Instruct-v0.2 with zero shot prompting.

```python
Task5.2+3.py [-h] --checkpoint {google/flan-t5-xxl,mistralai/Mistral-7B-Instruct-v0.2}

**arguments:**
--checkpoint: Specifies HF checkpoint to use for generation. One of "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"
**example:**
python Task5.2+3.py  --checkpoint google/flan-t5-xxl
```
The generation results from the FLAN model reported in report is saved in "data_tmp/5.3synthtic_scored.csv".

### Task 5.5 and 6 Training Scorer Model
To generate results in 5.5 and 5.6 using google-bert/bert-large-cased or FacebookAI/roberta-large.


```python
Task5.5+6.py [-h] --checkpoint {google-bert/bert-large-cased,FacebookAI/roberta-large}  --generation_csv PATH  [--keep_reference_summary]

**arguments:**
--generation_csv: CSV file with the generations from previous task, hint: saved in data_tmp/5.3synthtic_scored.csv, 
--checkpoint: Specifies HF checkpoint to use for the model for task 4.2. One of google-bert/bert-large-cased or FacebookAI/roberta-large.
--keep_reference_summary:  Whether to use use reference_summary as well.
**example:**
python Task5.5+6.py   --generation_csv data_tmp/5.3synthtic_scored.csv --checkpoint FacebookAI/roberta-large --keep_reference_summary
```
Four of the trained models reported in report are uploaded in HuggingFace model repository with ID: SushantGautam/roberta-large_synt_flan, 
SushantGautam/roberta-large_synt_flan_with_reference_summ, bert-large-cased_synt_flan, SushantGautam/bert-large-cased_synt_flan_with_reference_summ

 The training logs are published in Weight and Bias at https://wandb.ai/sushantgautam/nlp_assignment
 
### Task 5.7 Evaluation with Scorer Model
We can use same script as Task4.3-eval.py. And also use the same generations form Task 4, just with newer scorer model.
```python
Task4.3-eval.py [-h] --generation_csv PATH  --models ...

**arguments:**
--generation_csv: CSV file with the generations from previous task, hint: saved in data_tmp/
synthtic_scored.csv, 
--models: HF/local checkpoints of the models to use for scoring, like SushantGautam/roberta-large_synt_flan, Separate with space
--synth: Identifier to save the the scores in different file for synthetic data from task 5.7
**example:**
python Task4.3.py --synth --generation_csv data_tmp/4.3generations.csv --models  SushantGautam/roberta-large_synt_flan SushantGautam/roberta-large_synt_flan_with_reference_summ bert-large-cased_synt_flan SushantGautam/bert-large-cased_synt_flan_with_reference_summ
```
The script prints the metrics but also logs the intermediate scores for each candidate at data_tmp/5.7synthtic_scored.csv. 



### Task 6 Generation on blind set 
To generate final submission:

```python
Task6.py [-h] --checkpoint MODEL

**arguments:**
--checkpoint: Specifies HF/local checkpoint to use for the model, the best score was is SushantGautam/roberta-large_synt_flan
**example:**
python Task6.py --checkpoint SushantGautam/roberta-large_synt_flan
```

The candidate generation from the FLAN model as well as scoring with the best scorer model is saved in "data_tmp/6.FLAN_blind-candidate_scored.csv".
The summaries for submsission is saved at "final_submission.csv" as well as "final_submission.jsonl.gz" in the required format. 


## By:
**Sushant Gautam** and **Fernando Vallecillos Ruiz**


[Overleaf](https://www.overleaf.com/read/shpppdjvstgz#04eec5)
