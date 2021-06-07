

In order to evaluate BabyBERTa models on BLiMP, we require two steps:

1. Run `score.sh` using `BabyBERTa/blimp` as working directory. This will save pseudo-log-likelihoods to `blimp/output`

2. Run `accuracy.py` to printout accuracy scores.