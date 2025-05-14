# gen_err_sel_ssm

This repository contains the code for the experiments in the paper:  
**"Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention"**, submitted to NeurIPS 2025.

## Code Overview

- The Selective State-Space Model is implemented in PyTorch in  
  `models/selective_ssm.py`.

- The main training script is `main.py`, which loads data and trains the model.  
  You can run it from the command line, or use the helper script `run_main.sh` to run it.

## Experiments

- **Length-independence experiment:** `test_models.ipynb`  
- **Stability margin experiment:** `plot_s_A_T.ipynb`

This repository is anonymized for peer review.



