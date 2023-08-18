from gem_metrics.msttr import MSTTR
from gem_metrics.ngrams import NGramStats
from gem_metrics.texts import Predictions
import sys
from datasets import load_dataset, Dataset

def eval_model(gen_texts, output_reward_path=None):
    _msttr_metric = MSTTR(window_size=100)
    _n_gram_metric = NGramStats()
    predictions = Predictions(data={"filename": "", "values": gen_texts})
    diversity_metrics = {}
    msttr_metrics = _msttr_metric.compute(None, predictions)
    n_gram_metrics = _n_gram_metric.compute(None, predictions)
    for key, value in msttr_metrics.items():
        diversity_metrics[f"diversity_metrics/{key}"] = (None, value)
    for key, value in n_gram_metrics.items():
        diversity_metrics[f"diversity_metrics/{key}"] = (None, value)

    if output_reward_path is not None:
        with open(output_reward_path, mode='a') as fout:
            fout.write(str(diversity_metrics))
            fout.write("\n")



data_dir = sys.argv[1]
output_dir = sys.argv[2]

dataset = load_dataset("json", data_files=data_dir + "eval_set/my_eval_set.json", split="train", field="instances")


inputs = dataset['input']
outputs = dataset['output']
gen_texts = [inputs[i] + outputs[i][0] for i in range(len(inputs))]
eval_model(gen_texts, output_reward_path=output_dir)