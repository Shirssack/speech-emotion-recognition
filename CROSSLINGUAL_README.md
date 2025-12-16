# Layer-Wise Adversarial Disentanglement for Cross-Lingual Speech Emotion Recognition

Implementation of **HuBERT-based cross-lingual emotion recognition** with **layer-wise adversarial language disentanglement**.

## 🎯 Research Novelty

This implementation presents several novel contributions:

1. **First Application to HuBERT**: Layer-wise adversarial disentanglement applied to HuBERT for emotion recognition
2. **Multi-Layer Gradient Reversal**: Adversarial training at multiple transformer layers (not just final embedding)
3. **Hindi-English Cross-Lingual**: Focus on under-resourced language pair
4. **Real Data Only**: No synthetic data augmentation - purely real speech corpora

## 🏗️ Architecture

### Model Components

```
Input Audio
    ↓
HuBERT Feature Extractor (frozen)
    ↓
HuBERT Transformer Layers (1-12)
    ↓
[Layer 6] → Gradient Reversal → Language Discriminator
    ↓
[Layer 9] → Gradient Reversal → Language Discriminator
    ↓
[Layer 12] → Gradient Reversal → Language Discriminator
    ↓
Mean Pooling → Layer Norm
    ↓
Emotion Classifier
    ↓
Emotion Prediction
```

### Key Features

- **HuBERT Base**: 95M parameters, pretrained on LibriSpeech
- **Gradient Reversal Layer (GRL)**: Reverses gradients to confuse language discriminator
- **Layer-Wise Adversarial Training**: Applied at layers 6, 9, and 12 (customizable)
- **Language Discriminators**: Separate discriminators for each adversarial layer
- **Emotion Classifier**: 2-layer MLP on top of final representation

## 📊 Training Methodology

### Objective Function

```
L_total = L_emotion + λ_adv * (1/N) * Σ L_language_i
```

Where:
- `L_emotion`: Cross-entropy loss for emotion classification
- `L_language_i`: Cross-entropy loss for language discrimination at layer i
- `λ_adv`: Adversarial weight (default: 1.0)
- `N`: Number of adversarial layers

### Gradient Reversal

The gradient reversal layer implements:

```python
Forward:  y = x
Backward: ∂L/∂x = -α * ∂L/∂y
```

Where `α` controls the strength of adversarial training.

### Alpha Scheduling

Two scheduling strategies:

1. **Constant**: `α = 1.0` throughout training
2. **Progressive** (recommended): `α = 2/(1 + exp(-10*p)) - 1`
   - Where `p = epoch / total_epochs`
   - Gradually increases adversarial strength

## 🚀 Quick Start

### Installation

```bash
# Install additional dependencies
pip install transformers accelerate
```

### Training

**Basic Training** (all datasets):
```bash
python train_hubert_crosslingual.py --epochs 15 --batch_size 8
```

**Advanced Configuration**:
```bash
python train_hubert_crosslingual.py \
    --epochs 20 \
    --batch_size 8 \
    --lr 5e-5 \
    --lambda_adv 1.0 \
    --max_duration 5.0 \
    --adversarial_layers 6 9 12 \
    --alpha_schedule progressive
```

**English-Only Training** (for baseline):
```bash
python train_hubert_crosslingual.py \
    --train_csv data/csv/train_ravdess_4class.csv data/csv/train_tess_4class.csv \
    --val_csv data/csv/test_ravdess_4class.csv data/csv/test_tess_4class.csv
```

**Hindi-Only Training** (for baseline):
```bash
python train_hubert_crosslingual.py \
    --train_csv data/csv/train_hindi_4class.csv \
    --val_csv data/csv/test_hindi_4class.csv
```

### Evaluation

**Cross-Lingual Evaluation**:
```bash
python evaluate_crosslingual.py --model_path models/hubert_crosslingual_best.pt
```

This will evaluate on:
- English (in-domain)
- Hindi (in-domain)
- Combined (cross-lingual)

## 📈 Expected Results

### Performance Benchmarks

Based on the architecture and datasets:

| Training Scenario | English Acc | Hindi Acc | Cross-Lingual Gap |
|-------------------|-------------|-----------|-------------------|
| **English Only** | 75-85% | N/A | N/A |
| **Hindi Only** | N/A | 70-80% | N/A |
| **Multilingual (No Adv)** | 70-80% | 65-75% | ~10-15% |
| **Multilingual + Adversarial** | 75-85% | 70-80% | **~5-8%** ✓ |

The adversarial training should reduce the cross-lingual generalization gap by ~5-10%.

### What to Expect

**Good Signs**:
- Language discriminator accuracy ~50% (close to random = successful disentanglement)
- Similar performance on English and Hindi
- Smooth training curves

**Potential Issues**:
- Language discriminator accuracy ~100% = adversarial training not working
- Large gap between English and Hindi = poor cross-lingual transfer
- Unstable training = adjust `lambda_adv` or learning rate

## 🔧 Hyperparameter Tuning

### Critical Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lambda_adv` | 1.0 | 0.1-5.0 | Higher = stronger language disentanglement |
| `lr` | 1e-4 | 5e-5 to 2e-4 | Learning rate for AdamW |
| `adversarial_layers` | [6,9,12] | Any subset of 1-12 | Which layers to apply GRL |
| `alpha_schedule` | progressive | constant/progressive | How α changes over training |
| `batch_size` | 8 | 4-16 | Limited by GPU memory |
| `max_duration` | 5.0 | 3.0-8.0 | Longer = more context, more memory |

### Recommendations

**For Best Cross-Lingual Transfer**:
```python
lambda_adv = 1.5          # Strong adversarial
adversarial_layers = [6, 9, 12]  # Multi-layer
alpha_schedule = 'progressive'   # Gradual increase
```

**For Limited GPU Memory**:
```python
batch_size = 4
max_duration = 3.0
freeze_feature_extractor = True  # Already default
```

**For Faster Training**:
```python
adversarial_layers = [12]  # Only final layer
lambda_adv = 0.5          # Weaker adversarial
```

## 📝 Model Architecture Details

### HuBERT Layers

The 12 transformer layers learn hierarchical representations:

- **Layers 1-4**: Low-level acoustic features (phonemes, pitch)
- **Layers 5-8**: Mid-level features (prosody, speaker characteristics)
- **Layers 9-12**: High-level features (emotion, semantics)

**Why layers 6, 9, 12?**
- Layer 6: Mid-level (captures language-specific prosody)
- Layer 9: High-level (semantic/emotional content)
- Layer 12: Final representation (integrated features)

### Language Discriminator Architecture

Each discriminator:
```python
Input (768-dim)
  → Linear(768 → 256) + ReLU + Dropout(0.3)
  → Linear(256 → 128) + ReLU + Dropout(0.3)
  → Linear(128 → 2)  # English vs Hindi
  → Language Prediction
```

### Emotion Classifier Architecture

```python
Input (768-dim, pooled)
  → Dropout(0.3)
  → Linear(768 → 256) + ReLU
  → Dropout(0.3)
  → Linear(256 → 4)  # 4 emotions
  → Emotion Prediction
```

## 🎓 Research Applications

### Experimental Scenarios

1. **Baseline Comparison**
   ```bash
   # Train without adversarial (set lambda_adv=0)
   python train_hubert_crosslingual.py --lambda_adv 0.0
   ```

2. **Ablation Study: Number of Layers**
   ```bash
   # Single layer
   python train_hubert_crosslingual.py --adversarial_layers 12

   # Two layers
   python train_hubert_crosslingual.py --adversarial_layers 9 12

   # All layers (full)
   python train_hubert_crosslingual.py --adversarial_layers 6 9 12
   ```

3. **Ablation Study: Lambda**
   ```bash
   for lambda in 0.1 0.5 1.0 2.0 5.0; do
       python train_hubert_crosslingual.py --lambda_adv $lambda --save_dir models/lambda_$lambda
   done
   ```

4. **Alpha Schedule Comparison**
   ```bash
   # Constant alpha
   python train_hubert_crosslingual.py --alpha_schedule constant

   # Progressive alpha (recommended)
   python train_hubert_crosslingual.py --alpha_schedule progressive
   ```

### Analysis Scripts

**Visualize Layer Representations** (coming soon):
```python
from hubert_crosslingual_emotion import CrossLingualEmotionRecognizer

recognizer = CrossLingualEmotionRecognizer()
recognizer.load_model('models/hubert_crosslingual_best.pt')

# Extract layer outputs
_, _, layer_outputs = recognizer.model(audio, language_labels=None)

# Visualize with t-SNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

for layer_name, features in layer_outputs.items():
    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(features.cpu().numpy())
    plt.scatter(embedded[:, 0], embedded[:, 1])
    plt.title(f'{layer_name} Representation')
    plt.show()
```

## 💾 Model Files

After training, you'll have:

```
models/
├── hubert_crosslingual_best.pt      # Best validation model
└── hubert_crosslingual_final.pt     # Final epoch model
```

Model checkpoint includes:
- Model weights
- Optimizer state
- Training epoch
- Validation loss
- Emotion labels
- Adversarial layer configuration

## 🔍 Debugging Tips

### Issue: Language Discriminator Too Accurate

**Symptom**: Language discriminator >90% accuracy
**Cause**: Adversarial training not working
**Fix**:
- Increase `lambda_adv` (try 2.0 or 5.0)
- Use progressive alpha schedule
- Check gradient flow

### Issue: Poor Emotion Accuracy

**Symptom**: Emotion accuracy <60%
**Cause**: Too much adversarial confusion
**Fix**:
- Decrease `lambda_adv` (try 0.5 or 0.1)
- Use fewer adversarial layers
- Increase learning rate

### Issue: Training Unstable

**Symptom**: Loss oscillates wildly
**Cause**: Learning rate too high or lambda too high
**Fix**:
- Reduce learning rate (try 5e-5)
- Reduce `lambda_adv` (try 0.5)
- Use progressive alpha schedule
- Add gradient clipping (already implemented at 1.0)

### Issue: Out of Memory

**Fix**:
```bash
python train_hubert_crosslingual.py \
    --batch_size 4 \
    --max_duration 3.0
```

## 📚 Citation

If you use this implementation in your research:

```bibtex
@article{crosslingual_hubert_2024,
  title={Layer-Wise Adversarial Disentanglement for Cross-Lingual Speech Emotion Recognition using HuBERT},
  author={Your Name},
  year={2024},
  note={Implementation based on HuBERT and adversarial domain adaptation}
}
```

## 🤝 Related Work

This implementation builds upon:

1. **HuBERT**: Hsu et al. (2021) - "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
2. **Gradient Reversal**: Ganin & Lempitsky (2015) - "Unsupervised Domain Adaptation by Backpropagation"
3. **Cross-Lingual SER**: Various works on multilingual emotion recognition

## 🔗 Additional Resources

- [HuBERT Paper](https://arxiv.org/abs/2106.07447)
- [Gradient Reversal Paper](https://arxiv.org/abs/1409.7495)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 📊 Datasets Used

### RAVDESS
- **Language**: English
- **Speakers**: 24 actors
- **Emotions**: 4 classes (sad, neutral, happy, angry)
- **Samples**: ~672 utterances

### TESS
- **Language**: English
- **Speakers**: 2 actresses
- **Emotions**: 4 classes
- **Samples**: ~1,600 utterances

### Hindi Dataset
- **Language**: Hindi
- **Emotions**: 4 classes
- **Samples**: ~1,599 utterances

**Total**: ~3,871 samples across 2 languages

## 🎯 Future Enhancements

Potential improvements:

1. **More Languages**: Extend to other low-resource languages
2. **Contrastive Learning**: Add contrastive loss for better representations
3. **Attention Visualization**: Visualize which layers/tokens matter most
4. **Zero-Shot Transfer**: Test on completely unseen languages
5. **Speaker Adaptation**: Add speaker-adversarial training

---

**Questions?** See the main README.md or open an issue on GitHub.
