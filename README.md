# FLAG 2027 1st-Place System Reproduction Repository (Team `susupro`)

Official training and trial-independent inference repository for **Team `susupro`** (Representative: **Yan Su**, `hi@yansu.me` / `y-su24@mails.tsinghua.edu.cn`) in the **IEEE ICASSP 2027 Face-Voice Association across LAnguages and Gender (FLAG 2027) Grand Challenge**.

- **CodaBench Evaluation Submission**: `#1057629` / `#1057627`
- **Official Final Evaluation EER**: **`2.13%` displayed (`2.129763%` exact mean EER)**
  - `no_gender / English`: **`1.530612%`** (`392` trials)
  - `no_gender / Bangla`: **`1.745636%`** (`802` trials)
  - `gender / English`: **`3.846154%`** (`416` trials)
  - `gender / Bangla`: **`1.396648%`** (`716` trials)
- **Strict Trial Independence**: `singleton_max_diff = 0.0` and `permutation_max_diff = 0.0` across all `2,326` evaluation trials.

## 1. Quick Start: 1-Command Inference Reproduction (< 2 Seconds)

Compatible with Python 3.8 to 3.13 on Linux and macOS (CPU-only, `< 110 MiB` RAM):

```bash
python -m pip install -r requirements.txt
python run.py
python verify_submission.py --submission output/submission.zip
```

Or run in an isolated offline `linux/amd64` container:

```bash
docker build --platform linux/amd64 -t flag2027-susupro .
docker run --rm --network none --platform linux/amd64 --memory 4g --cpus 1 \
  -v "$PWD/output:/output" flag2027-susupro --output-dir /output
```

### Inference Output & Verification
- Generates `output/submission.zip` (4 protocol/language score files in official `<pair_id> <distance>` format, where lower score indicates stronger same-identity association) and `output/status.json`.
- Reproduces CodaBench submission `#1057629` (`SHA256: 69a660f98b09fe1c7071a01329b87769b1972e548d5db59269d2e7e82eeee8bf`) with 100% identical trial rankings across macOS and `linux/amd64` (max floating-point difference $4.44 \times 10^{-13}$).
- `run.py` automatically verifies both **singleton independence** (scoring each trial individually with `batch_size=1` vs. full list) and **permutation invariance** (scoring a randomly shuffled trial list vs. original order), confirming `singleton_max_diff == 0.0` and `permutation_max_diff == 0.0` on every cell.

## 2. Repository Structure & Complete Pipeline

```text
.
├── run.py                         # Trial-independent evaluation inference entry point
├── core.py                        # Training-standardized feature alignment & conditional bilinear forward pass
├── extract_features.py            # Deterministic feature extraction, fixed-slice audio averaging & template correlation
├── train.py                       # 5-fold speaker-disjoint & Leave-Language-Out (Urdu/Hindi) conditional bilinear trainer
├── train_network.py               # Utterance-level cross-modal dual-tower trainer (ConeTower, MSM, PAEFF, ConvLoRA, VER)
├── models.py                      # PyTorch neural architectures for cross-modal projection & pair discrimination
├── model_inference.py             # Deterministic NumPy/PyTorch forward execution for trained dual-tower graphs
├── verify_submission.py           # Official FLAG 2027 submission format & trial-ID coverage validator
├── configs/
│   ├── flag2027_config.yaml       # Full hyperparameter, fold-split, and preprocessing configuration
│   ├── encoder_inventory.json     # Pretrained face, voice, and demographic encoder specifications
│   └── encoder_exports.example.json
├── assets/
│   ├── *-head.npz                 # Fixed training-fitted conditional bilinear fusion heads (4 protocol/language cells)
│   └── *-indep-input.npz          # Pre-extracted per-trial distance & conditioning tensors for 2-second reproduction
├── system_description_2p.pdf      # 2-Page IEEE ICASSP 2027 System Description paper
├── questionnaire_completed.pdf    # Completed System Description & Compliance Questionnaire (PDF/DOCX/MD)
├── requirements.txt               # Runtime dependencies (numpy, scipy)
├── requirements-training.txt      # Training dependencies (torch)
└── Dockerfile                     # Offline linux/amd64 reproducibility container specification
```

## 3. Feature Extraction & Training Pipeline

### Step 1: Deterministic Feature Extraction & Fixed-Slice Audio Averaging (`extract_features.py`)
All pretrained face and audio backbones (listed in `configs/encoder_inventory.json`) are evaluated in deterministic inference mode (`eval()`, parameter gradients disabled). For variable-length audio, `extract_features.py` extracts $K=3$ uniformly spaced $6.0\,\text{s}$ sub-slices at $16\,\text{kHz}$, $\ell_2$-normalizes each sub-slice embedding, averages them, and re-normalizes (strictly compliant with FAQ #13). It also computes the trial-independent training-template spectral response correlation ($\rho_{\text{relg}}$) against fixed South Asian training speaker centroids.

```bash
# Verify deterministic fixed-slice audio segmentation & template spectral correlation
python extract_features.py --smoke-test

# Extract deterministic embedding from a registered TorchScript encoder checkpoint
python extract_features.py --manifest configs/encoder_exports.example.json \
  --encoder voxb2_r293 --input sample_waveform.npy --output output/embedding.npy
```

### Step 2: Cross-Modal Dual-Tower Metric & Pair Discrimination Training (`train.py --stage network`)
Trains the five cross-modal neural projection families (`cone`, `msm`, `paeff`, `convlora`, `ver`) on non-Bengali utterance banks (official `v4` English training set + non-Bengali `VoxCeleb2` + `MAV-Celeb v1` English/Urdu & `v2` English/Hindi) with feature-level dropout ($p \in [0.30, 0.91]$), cross-recording same-identity positive sampling, and same-gender hard negative mining:

```bash
python -m pip install -r requirements-training.txt

# Quick end-to-end verification of the neural trainer
python train.py --stage network --smoke-test --arch ver --steps 2 --held-fold 0

# Full utterance-bank training for a specific architecture and fold
python train.py --stage network --bank utterances.npz --arch msm --steps 1200 --held-fold 0 --output-dir output/msm-fold0
```

### Step 3: Conditional Bilinear Fusion Head Training (`train.py`)
Fits the second-order conditional bilinear fusion head (`VER_FUSE_X`, combining standardized per-trial distance streams $\tilde{x}$, demographic/acoustic conditioning variables $\tilde{c}$, availability indicators $o_{\mathcal{A}}$, and outer-product interactions $\tilde{x} \otimes g$) via **5-Fold Speaker-Disjoint Cross-Validation** on the 70 official training identities and **Leave-Language-Out (`English -> Urdu/Hindi`)** validation on `MAV-Celeb v1/v2`:

```bash
# Quick end-to-end verification of the 5-fold speaker-disjoint fusion trainer
python train.py --smoke-test

# Fit conditional bilinear head on training folds + Urdu/Hindi Leave-Language-Out validation
python train.py --train-data train_streams.npz --validation-data mav_urdu_hindi.npz --output-dir output/fitted-head
```

## 4. Compliance Summary

1. **Zero Test-Set Training or Transductive Processing (FAQ #1 to #3)**: No clustering, kNN, graph construction, pseudo-labeling, or test-set statistics are computed across progress or evaluation samples. Every trial $(f_i, v_i)$ is scored strictly independently (`singleton_max_diff == 0.0`, `permutation_max_diff == 0.0`).
2. **Zero Post-Hoc Score or Cohort Normalization (FAQ #4)**: No Z-Norm, T-Norm, S-Norm, AS-Norm, or CSLS cohort normalization is applied at test time. Input streams are standardized solely using fixed mean and standard deviation vectors estimated on the training set.
3. **Zero Bengali Audio-Encoder Adaptation (FAQ #6 to #9)**: All pretrained audio encoders (`WeSpeaker VoxBlink2 ResNet293`, `SimAM-ResNet100`, `ReDimNet-B2/v2`, `WavLM-Large`, `ECAPA-TDNN`, `audEERING`) are used strictly frozen. Any speaker predicted as Bengali in external corpora was completely excluded at the speaker level.
4. **Frozen Demographic Consistency & Conditioning Gates (FAQ #10 to #12)**: Frozen age and gender predictors provide fixed demographic consistency terms and bilinear conditioning gates, with all weights fixed on training folds prior to evaluation.
