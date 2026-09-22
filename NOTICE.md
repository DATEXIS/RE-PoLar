# NOTICE

Third-party data and code this repository depends on, and their licenses.
All code under `re_polar/` is original to this project unless noted below.
PoLar's own repository carries no license, so `re_polar/core`, `re_polar/mcts`, and
`re_polar/router` are independent reimplementations from the published paper's
formalization, not derived from PoLar's code.

## dart-math (`re_polar/vendor/dart_math/`)

Answer-equivalence grading for math answers. Source: [hkust-nlp/dart-math](https://github.com/hkust-nlp/dart-math),
MIT License.

## hkust-nlp/dart-math-pool-{math,gsm8k}(-query-info)

`re_polar/datasets/dart_math.py` loads these four datasets from the
Hugging Face Hub (Tong et al., NeurIPS'24, arXiv:2407.13690) to reconstruct
question text, ground-truth answers, and per-query difficulty. MIT
License.

## TIGER-Lab/MMLU-Pro

`re_polar/datasets/mmlu_pro_domains.py` and `re_polar/datasets/mmlu_pro_hf.py` load
[TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
from the Hugging Face Hub. MIT License.

## EleutherAI/asdiv

`re_polar/datasets/asdiv.py` loads
[EleutherAI/asdiv](https://huggingface.co/datasets/EleutherAI/asdiv) from
the Hugging Face Hub. CC-BY-NC-4.0. Corpus: Miao et al., 2020
(arXiv:2106.15772).

## MU-NLPC/Calc-mawps

`re_polar/datasets/mawps.py` loads
[MU-NLPC/Calc-mawps](https://huggingface.co/datasets/MU-NLPC/Calc-mawps)
(test split) from the Hugging Face Hub. MIT. Cited via Kadlčík et al., 2023
(Calc-X/Calcformers, EMNLP 2023).
