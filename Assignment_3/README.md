# Assignment 3
## Fox Modules used:
```
nlpl-llmtools/03-foss-2022b-Python-3.10.8
nlpl-transformers/4.38.2-foss-2022b-Python-3.10
```

## Training and evaluation
### Task 3.2 and 3.3
To generate results in 3.2 and 3.3 using **facebook/opt-6.7b** or **mistralai/Mistral-7B-v0.1** with zero and few shot prompting.

```
train_model.py [-h] --checkpoint {facebook/opt-6.7b,mistralai/Mistral-7B-v0.1} [--multi_shot]

**arguments:**
--checkpoint: Specifies HF checkpoint to use for the model. One of facebook/opt-6.7b or mistralai/Mistral-7B-v0.1.
--multi_shot: Enables the use of multiple prompts for each article. This option does not require a value. Dont include for zero shot prompting.
```

## By:



[Overleaf](https://www.overleaf.com/read/shpppdjvstgz#04eec5)
