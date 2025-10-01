# gen_err_sel_ssm

This repository contains the code for the experiments in the paper **"Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention"**, Accepted as a poster to [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117416).

## 🔧 Setup (Conda)

To create the environment:

```bash
conda env create -f environment.yml -n gen_err_sel_ssm
conda activate gen_err_sel_ssm
```

## 📦 Code Overview

- The Selective State-Space Model is implemented in PyTorch in `models/selective_ssm.py`.
- The main script is `main.py`, which loads data, trains the model, and saves the results for different experiments.
  You can run it directly from the command line, or modify and execute the helper script:

```bash
bash run_main.sh
```

## 📊 Experiments

- **Length-independence experiment:** `test_models.ipynb`  
- **Stability margin experiment:** `plot_sA_T.ipynb`
- **Significant analysis:** `significance_test.ipynb`

---

## 🗂️ Datasets

- **IMDB (sentiment, 25k train / 25k test, binary labels):** [Stanford IMDB (Maas et al., 2011)](https://ai.stanford.edu/~amaas/data/sentiment/) • [HF dataset card](https://huggingface.co/datasets/stanfordnlp/imdb)
- **ListOps (hierarchical parsing / long-context reasoning):** part of **Long Range Arena (LRA)** — [arXiv](https://arxiv.org/abs/2011.04006) • [LRA GitHub](https://github.com/google-research/long-range-arena)
- **Majority (synthetic):** binary sequence where the label equals the majority bit; we use a simple generator released in this repo.

Dataset loading functions are in `training/load_datasets.py`.

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{honarpisheh2025generalization,
  title={Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention},
  author={Honarpisheh, Arya and Bozdag, Mustafa and Camps, Octavia and Sznaier, Mario},
  journal={arXiv preprint arXiv:2502.01473},
  year={2025}
}



