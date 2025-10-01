import torch
from eval_metrics import get_ece, get_brier, get_auroc
from scipy.stats import pearsonr, spearmanr
import json
import argparse

if __name__ == "__main__":
    # get model name
    parser = argparse.ArgumentParser()
    model_names = [
        'Qwen/Qwen3-8B',
        'google/gemma-3-4b-it',
    ]
    parser.add_argument('--model_name', type=str, default=model_names[1], choices=model_names, help='huggingface model name')
    args = parser.parse_args()
    model_name = args.model_name
    half_model_name = model_name.split('/')[1]
    results_fstem = f'results/factscore/{half_model_name}'

    metrics = [get_ece, get_brier, get_auroc]
    metric_params = {
        get_ece: {'n_bin': 10, 'return_breakdown': False},
        get_brier: {},
        get_auroc: {},
    }

    # load scores
    decisions = []
    with open(f'factscore_claims/{half_model_name}.jsonl') as f:
        decomps = [json.loads(line) for line in f.readlines() if line.strip()]
    scores = torch.tensor([claim['is_supported'] for decomp in decomps for claim in decomp['decisions']]).float()
    print(f'Accuracy {scores.float().mean()} over {len(scores)} claims')
    print()
    # compute entity score: mean of claim scores
    entity_scores = torch.tensor([torch.mean(torch.tensor([claim['is_supported'] for claim in decomp['decisions']]).float()) for decomp in decomps])

    # load confidences
    dinco_confs = torch.load(f'{results_fstem}/dinco_confs.pth')
    # compute entity confidence: mean of claim confidences
    dinco_entity_confs = torch.tensor([torch.mean(decomp_confs[decomp_confs >= 0]) for decomp_confs in dinco_confs])
    dinco_confs = dinco_confs[dinco_confs >= 0]

    # compute calibration metrics
    for metric_i, metric in enumerate(metrics):
        print(metric.__name__, metric(scores, dinco_confs, **metric_params[metric]))

    # compute entity-level calibration metrics
    print('pearson', pearsonr(entity_scores.numpy(), dinco_entity_confs.numpy()).statistic.item())
    print('spearman', spearmanr(entity_scores.numpy(), dinco_entity_confs.numpy()).correlation.item())
