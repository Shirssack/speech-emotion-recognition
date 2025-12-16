# Adversarial HuBERT: Train + Predict Cheat Sheet

Follow this guide to train the layer-wise adversarial HuBERT model and generate predictions from the trained checkpoint.

## 1) Environment prep
- Python 3.8+ recommended.
- Install dependencies: `pip install -r requirements.txt`
- Ensure the CSV splits exist (English + Hindi):
  ```bash
  ls data/csv/train_*_4class.csv
  ls data/csv/test_*_4class.csv
  ```

## 2) Sanity-check the architecture
Before a long run, verify the model and loss wiring:
```bash
python adversarial_hubert_emotion.py
```
You should see the model summary, parameter counts, and a quick forward/loss test.

## 3) Train the adversarial model
Use the default recipe (saves to `models/adversarial_hubert/default/`):
```bash
python train_adversarial_hubert.py \
    --epochs 20 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --output_dir models/adversarial_hubert \
    --experiment_name default
```
Quick one-liner version if you just need the base run:
```bash
python train_adversarial_hubert.py --epochs 20 --batch_size 8 --learning_rate 3e-5 --output_dir models/adversarial_hubert --experiment_name default
```
Helpful tweaks:
- Lower memory footprint (e.g., 4–6GB GPUs): add `--batch_size 4 --adversarial_layers 6 12 --hidden_dim 128 --max_duration 4.0 --gradient_checkpointing`.
- Stronger adversarial pressure: increase `--language_weight` (e.g., `0.3`) or train longer (e.g., `--epochs 25`).

Artifacts to expect:
- `models/adversarial_hubert/<experiment>/best_model.pth` — best checkpoint by validation score.
- `training_log.csv` inside the experiment folder — metrics per epoch.

## 4) Evaluate and get predictions
Run cross-lingual evaluation with the saved checkpoint (also produces per-language confusion matrices and a JSON summary):
```bash
python evaluate_crosslingual.py \
    --model_path models/adversarial_hubert/default/best_model.pth \
    --output_dir results/crosslingual/default \
    --evaluate_both_directions
```
Outputs include:
- `results/crosslingual/<experiment>/crosslingual_evaluation.json` — key metrics (within-language F1, cross-lingual F1, transfer gap, language discriminator accuracy).
- PNG confusion matrices for English/Hindi (within-language and cross-lingual).

## 5) Using the model for quick predictions
- For dataset-level predictions, reuse `evaluate_crosslingual.py` with a custom `--output_dir` (the script runs inference over the test splits and saves metrics + visualizations).
- For one-off sanity checks after training, re-run `python adversarial_hubert_emotion.py --model_path <checkpoint>` to load the saved weights and ensure forward passes still work.

Tip: Good runs typically show a cross-lingual transfer gap under 10% and language discriminator accuracy near 50–60% (indicating language-invariant features).
