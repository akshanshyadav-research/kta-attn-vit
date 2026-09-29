"""Memory-bounded Kendall scores across embedding features, not spatial ranks."""
import torch


def kendall_scores(cls_token, tokens, variant='notebook', chunk_size=16):
    if variant not in ('notebook', 'tau-b'):
        raise ValueError('variant must be notebook or tau-b')
    if tokens.shape[-1] < 2 or chunk_size < 1:
        raise ValueError('Need at least two features and positive chunk size')
    i, j = torch.triu_indices(tokens.shape[-1], tokens.shape[-1], 1, device=tokens.device)
    a = torch.sign(cls_token[:, i] - cls_token[:, j]).unsqueeze(1)
    result = []
    for patches in tokens.split(chunk_size, dim=1):
        b = torch.sign(patches[..., i] - patches[..., j])
        numerator = (a * b).sum(-1)
        if variant == 'notebook':
            score = numerator / i.numel()
        else:
            denominator = ((a != 0).sum(-1) * (b != 0).sum(-1)).float().sqrt()
            # All-tied inputs have undefined tau-b; neutral score is our explicit policy.
            score = torch.where(denominator > 0, numerator / denominator, 0)
        result.append(score)
    return torch.cat(result, dim=1)
