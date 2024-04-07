import evaluate
import numpy as np

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def compute_metrics(predictions, references):
    rouge_result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en", rescale_with_baseline=True)

    prediction_lens = [len(e.split()) for e in predictions]
    metrics_result = {"gen_len": np.mean(prediction_lens)}

    for key, value in rouge_result.items():
        metrics_result[f"{key}"] = round(value, 4)
    for key, value in bertscore_result.items():
        if key in ["precision", "recall", "f1"]:
            metrics_result[f"bert-score-{key}"] = np.mean(value).item()
    return metrics_result

import glob, json
import pandas as pd


def json_folder_to_df_prompt_experiments(path):
    all_jsons = glob.glob(path+"/**.json")
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
                json_data_list.append([i[0], i[1], i[2].strip()])
    return pd.DataFrame(json_data_list, columns=["org_row_ID", "prompt#", "summary"])
