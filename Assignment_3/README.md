# Assignment 3
## Fox Modules used:
```
nlpl-llmtools/03-foss-2022b-Python-3.10.8
nlpl-transformers/4.38.2-foss-2022b-Python-3.10
nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
```

## Training and evaluation
### Task 3.2 and 3.3
To generate results in 3.1:
```
TODO
```

### Task 3.2 and 3.3
To generate results in 3.2 and 3.3 using **facebook/opt-6.7b** or **mistralai/Mistral-7B-v0.1** with zero and few shot prompting.

```
Task3.2+3.py [-h] --checkpoint {facebook/opt-6.7b,mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.2,google/flan-t5-xxl} [--multi_shot]

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model. One of facebook/opt-6.7b or mistralai/Mistral-7B-v0.1.
--multi_shot: Enables the use of multiple prompts for each article foir task 3.2.2. This option does not require a value. Dont include for zero shot prompting for 3.2.1. Doesn't apply when using mistralai/Mistral-7B-Instruct-v0.2 and google/flan-t5-xxl which only is implemneted for zero-shot for 3.3.
```

### Task 4.1 and 4.2
To generate results in 4.1 and 4.2 using google-bert/bert-large-cased or FacebookAI/roberta-large with zero and few shot prompting.

```
Task4.1+2.py [-h] --checkpoint {google-bert/bert-large-cased,FacebookAI/roberta-large} --correctness_metric {overall,coherence,accuracy,coverage}

**arguments:**
--correctness_metric: Metric(s) to be used for correctness score for task 4.1. Separated by space. 
--checkpoint: Specifies HF checkpoint to use for the model for task 4.2. One of google-bert/bert-large-cased or FacebookAI/roberta-large.
**example:**
python Task4.1+2.py --checkpoint FacebookAI/roberta-large  --correctness_metric accuracy coherence
```

### Task 4.3
To generate results in 4.3 using google/flan-t5-xxl or mistralai/Mistral-7B-Instruct-v0.2 with zero shot prompting.

```
Task4.3.py [-h] --checkpoint {google/flan-t5-xxl,mistralai/Mistral-7B-Instruct-v0.2} --correctness_metric {overall,coherence,accuracy,coverage}

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model for task 4.2. One of "google/flan-t5-xxl", "mistralai/Mistral-7B-Instruct-v0.2"
python Task4.3.py --checkpoint google/flan-t5-xxl
```
The results from the FLAN model reported in report is saved in "data_tmp/4.3generations.csv".

## By:



[Overleaf](https://www.overleaf.com/read/shpppdjvstgz#04eec5)
