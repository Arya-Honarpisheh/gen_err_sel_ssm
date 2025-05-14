# gen_err_sel_ssm

This repository contains the code for the experiments in the paper:  
**"Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention"**, submitted to NeurIPS 2025.

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

---

This repository is anonymized for peer review.
