# EMGFF
Selected code and recording-level five-fold splits for EMGFF

This repository provides the main model implementation in PyTorch and the recording-level five-fold splits for DeepShip and ShipsEar.

- `model.py`: Main model, including Mel/MATS encoders, multi-scale fusion, DEMON-guided channel gating, Transformer, classification heads, center loss, and a forward-pass check.
- `deepship.csv`: Recording-level five-fold assignments for DeepShip.
- `shipsear.csv`: Recording-level five-fold assignments for ShipsEar.
- `README.md`: Repository overview and basic usage notes.

The CSV files specify the individual recordings assigned to each of the five folds. All segments from the same recording should follow its fold assignment.

The model takes preprocessed features of shape `(B, 3, F, T)` in **Mel, DEMON, MATS** order and returns feature vectors, main classification logits, and auxiliary DEMON logits. Mel and MATS must be aligned to shared physical frequency bands during preprocessing.

Dependencies: `torch` and `einops`. Run `python model.py` for a forward-pass check. Before creating an optimizer or loading weights, run a dummy forward pass with the actual feature dimensions to initialize the classification heads.

Raw audio, preprocessing and training/evaluation scripts, trained weights, and ablation models are not included.
