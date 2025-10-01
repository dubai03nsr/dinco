import torch
from sklearn.metrics import roc_auc_score

def get_ece(scores, probs, n_bin=10, return_breakdown=False):
    probs = torch.clamp(probs, 0, 1)

    bounds = torch.linspace(0, 1, n_bin + 1)
    bounds[-1] = 1.1 # make last bin include 1

    ece = torch.tensor(0.)
    if return_breakdown:
        accs = torch.zeros(n_bin)
        confs = torch.zeros(n_bin)
        sizes = torch.zeros(n_bin, dtype=torch.long)

    for bin_i in range(n_bin):
        mask = torch.logical_and(probs >= bounds[bin_i], probs < bounds[bin_i + 1])
        if mask.sum() == 0:
            continue

        acc = scores[mask].mean()
        conf = probs[mask].mean()
        ece += mask.float().mean() * torch.abs(acc - conf)

        if return_breakdown:
            accs[bin_i] = acc
            confs[bin_i] = conf
            sizes[bin_i] = mask.sum()

    if return_breakdown:
        return ece.item(), accs, confs, sizes
    else:
        return ece.item()

def get_brier(scores, probs):
    return torch.mean((scores - probs) ** 2).item()

def get_auroc(scores, confs):
    return roc_auc_score(scores.numpy(), confs.numpy())