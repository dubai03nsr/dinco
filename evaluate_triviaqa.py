"""
Evaluate calibration metrics for TriviaQA
"""

import torch
from eval_metrics import get_auroc, get_ece, get_brier
import argparse

if __name__ == "__main__":
    # get model name
    parser = argparse.ArgumentParser()
    model_names = [
        'Qwen/Qwen3-8B',
        'Qwen/Qwen3-1.7B',
        'meta-llama/Llama-3.2-3B-Instruct',
        'google/gemma-3-4b-it',
    ]
    parser.add_argument('--model_name', type=str, default=model_names[1], choices=model_names, help='huggingface model name')
    args = parser.parse_args()
    model_name = args.model_name
    half_model_name = model_name.split('/')[1]
    results_fstem = f'results/triviaqa/{half_model_name}'

    metrics = [get_ece, get_brier, get_auroc]
    metric_params = {
        get_ece: {'n_bin': 10, 'return_breakdown': False},
        get_brier: {},
        get_auroc: {},
    }

    # load scores
    judge_probs = torch.load(f'{results_fstem}/judge-probs.pt')
    scores = (judge_probs[:, 0] > judge_probs[:, 1]).float() # Yes > No
    print('Accuracy', scores.float().mean().item())
    print()

    # load confidences
    dinco_confs = torch.load(f'{results_fstem}/dinco_confs.pth')

    # compute calibration metrics
    for metric_i, metric in enumerate(metrics):
        print(metric.__name__, metric(scores, dinco_confs, **metric_params[metric]))
