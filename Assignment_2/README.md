# Assignment 2
### Fox Modules used:
```
nlpl-nlptools/01-foss-2022b-Python-3.10.8
nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
```

### Training and evaluation

train_cli.py allows you to train and test models with specified configurations for multi-language tasks. It supports both fine-tuning and freezing the transformer layers. Additionally, it provides hyperparameter variations as well as an option to separate validation for each language for Task 2 and saves the plots accordingly.


```python
python train_cli.py --source_langs lang1 lang2 ... --target_langs lang1 lang2 ... --model_name model_path --batch_size batch_size --epoch epoch --learning_rate learning_rate --finetune --separate_val

--source_langs: Source languages for training, separated by space (required; example: en de).
--target_langs: Target languages for evaluation, separated by space (required; example: en de).
--model_name: Model name or path (required; example: "/fp/projects01/ec30/models/xlm-roberta-base/").
--batch_size: Batch size for training (default: 64).
--epoch: Number of epochs for training (default: 20).
--learning_rate: Learning rate for training (default: 1e-5).
--finetune: Flag to finetune the model. If not set, the transformer layers will be frozen.
--separate_val: Flag to use separated validation for each language for Task 2. Also saves the plots. (recommended)

Example:
python train_cli.py --source_langs en de --target_langs fr sw  --model_name /fp/projects01/ec30/models/xlm-roberta-base/ --batch_size 64 --epoch 20 --learning_rate 1e-5 --finetune --separate_val

```

### Surprise Language

The output TSV is saved as **pred_tags_mapped.tsv**.
The best model is saved at '/cluster/work/projects/ec30/fernavr/best_model'

To reproduce, run the evaluation.py script:

```python
python evaluation.py --model /cluster/work/projects/ec30/fernavr/best_model --data /fp/projects01/ec30/IN5550/obligatories/2/surprise/surprise_test_set.tsv

Documentation:
python evaluation.py --model /path/to/model/folder --data /path/to/data/file.tsv

--model, -m: Path to the folder containing the pre-trained model. Defaults to '/cluster/work/projects/ec30/fernavr/best_model'.
--data, -d: Path to the .tsv file with one column for data input. Defaults to '/fp/projects01/ec30/IN5550/obligatories/2/surprise/surprise_test_set.tsv'.
```

### Results Reported
All then plots and training logs are stored in **results** directory.


## By:
**Fernando Vallecillos Ruiz** and **Sushant Gautam**

[Overleaf](https://www.overleaf.com/read/dcspmfcnztbp#3a9cdd)
