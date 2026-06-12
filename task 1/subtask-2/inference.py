import torch
from collections import Counter
import math

def load_checkpoint(model, experiment_name, device):
    path = f"checkpoints/{experiment_name}_best.pt"
    model.load_state_dict(torch.load(path, map_location=device)) # map_location=device assures remapping of the weights to the correct device
    model.eval()
    print(f"loaded checkpoint: {path}")
    return model


# METRICS

def exact_match(references, hypotheses):
    assert len(references) == len(hypotheses)
    return sum(r == h for r, h in zip(references, hypotheses)) / len(references)

# ---------------------------------------------------------------------------

def edit_distance(ref, hyp): # the minimum number of insertions, deletions, and substitutions to turn the hypothesis into the reference
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)] 

    for i in range(m + 1): dp[i][0] = i # to convert first i chars of ref to first 0 chars of hyp, we need i deletions
    for j in range(n + 1): dp[0][j] = j # to convert first 0 chars of ref to first i chars of hyp, we need i insertions

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1] # if same char, then no need to insert/delete
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) # deletion, insertion, replacement

    return dp[m][n]

def mean_edit_distance(references, hypotheses):
    return sum(edit_distance(r, h) for r, h in zip(references, hypotheses)) / len(references)

# ---------------------------------------------------------------------------

def bleu_score(references, hypotheses):
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def clipped_precision(refs, hyps, n): # to prevent cheating by repeating words
        total_correct = 0
        total_count = 0

        for ref, hyp in zip(refs, hyps):
            ref_ngrams = Counter(ngrams(ref, n))
            hyp_ngrams = Counter(ngrams(hyp, n))

            for gram, count in hyp_ngrams.items():
                total_correct += min(count, ref_ngrams.get(gram, 0)) # smallest of word appearances in ref, hyp
            total_count += sum(hyp_ngrams.values()) # sum for all words

        return total_correct / total_count

    # brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    bp = 1 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    precisions = [clipped_precision(references, hypotheses, n) for n in range(1, 5)]

    if any(p == 0 for p in precisions): # to get rid of log(0) issue
        return 0.0
    
    # GM of n-gram precisions (not AM because we want to penalize for having a precision of 0)
    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4) # using logs for numerical stability
    return round(score, 4)

@torch.no_grad()
def evaluate(model, loader, vocab, device):
    model.eval()
    references = []
    hypotheses = []

    for buggy, fixed in loader:
        buggy = buggy.to(device)

        for i in range(buggy.shape[0]):
            ref = vocab.decode([t for t in fixed[i].tolist() if t not in (0, 1, 2)])
            hyp = vocab.decode(model.generate(buggy[i]))

            references.append(ref)
            hypotheses.append(hyp)

    return {
        "bleu": bleu_score(references, hypotheses),
        "exact_match": exact_match(references, hypotheses),
        "mean_edit_dist": round(mean_edit_distance(references, hypotheses), 2),
    }

@torch.no_grad()
def show_examples(model, loader, vocab, device, n=5):
    model.eval()
    count = 0

    for buggy, fixed in loader:
        buggy = buggy.to(device)

        for i in range(buggy.shape[0]):
            if count >= n:
                return

            src = vocab.decode([t for t in buggy[i].tolist() if t not in (0, 1, 2)])
            ref = vocab.decode([t for t in fixed[i].tolist() if t not in (0, 1, 2)])
            hyp = vocab.decode(model.generate(buggy[i]))

            print(f"\n{'─' * 60}")
            print(f"BUGGY:     {' '.join(src)}")
            print(f"PREDICTED: {' '.join(hyp)}")
            print(f"REFERENCE: {' '.join(ref)}")
            print(f"MATCH:     {'YES' if hyp == ref else 'NO'}")
            count += 1