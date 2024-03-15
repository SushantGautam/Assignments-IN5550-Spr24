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

--source_langs: List of source languages for training (required).
--target_langs: List of target languages for training and evaluation (required).
--model_name: Model name or path (required).
--batch_size: Batch size for training (default: 64).
--epoch: Number of epochs for training (default: 20).
--learning_rate: Learning rate for training (default: 1e-5).
--finetune: Flag to finetune the model. If not set, the transformer layers will be frozen.
--separate_val: Flag to use separated validation for each language for Task 2. Also saves the plots. (recommended)
```

### Surprise Language
TODO

## By:
**Fernando Vallecillos Ruiz** and **Sushant Gautam**

[Overleaf](https://www.overleaf.com/read/dcspmfcnztbp#3a9cdd)
