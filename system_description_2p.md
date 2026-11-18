# Trial-Independent Cross-Modal Metric Learning and Conditional Bilinear Fusion for Face-Voice Association Across Languages and Gender

**Author**: Yan Su  
**Affiliation**: Independent Researcher (`hi@yansu.me`, `y-su24@mails.tsinghua.edu.cn`, CodaBench Team: `susupro`)

## Abstract
We present the 1st-place system (**Team `susupro`**) for the IEEE ICASSP 2027 Face-Voice Association across LAnguages and Gender (FLAG 2027) Grand Challenge. Addressing severe embedding-cone anisotropy, unseen Bengali acoustic drift, and demographic shortcut collapse in the same-gender (`gender`) protocol under strict single-trial independence constraints, our framework (**VER-FUSE-X**) integrates three synergistic pillars: (1) deterministic representation extraction across 15 frozen pretrained facial, speaker, and paralinguistic backbones with fixed-duration audio sub-slice averaging; (2) five complementary high-dropout cross-modal dual-tower metric networks (`ConeTower`, `MSM`, `PAEFF`, `ConvLoRA`, `VER_MLP`) co-trained on non-Bengali speakers alongside a trial-independent South Asian training-template spectral response correlation operator ($\rho_{\text{relg}}$); and (3) a second-order conditional bilinear fusion head that dynamically gates 87 (`no_gender`) or 114 (`gender`) standardized cross-modal distance streams with 16 frozen demographic and acoustic conditioning variables. Evaluated strictly one trial at a time (`singleton_max_diff = 0.0`, `permutation_max_diff = 0.0`) with zero post-hoc cohort normalization and zero Bengali audio-encoder adaptation, our system achieves an official mean Equal Error Rate (EER) of **2.13%** (**2.129763%** exact mean: **1.530612%** on `no_gender/English`, **1.745636%** on `no_gender/Bangla`, **3.846154%** on `gender/English`, and **1.396648%** on `gender/Bangla`), ranking **1st overall** among 45 teams.

**Keywords**: Face-voice association, cross-modal metric learning, conditional bilinear fusion, speaker verification, ICASSP 2027 FLAG Challenge

## 1. Introduction & Problem Formulation
Cross-modal face-voice association requires determining whether a static facial image $f_i$ and a speech segment $v_i$ originate from the same biometric identity across heard (`English`) and unheard (`Bangla`) languages under both unrestricted (`no_gender`) and same-gender negative (`gender`) evaluation protocols. Empirical diagnosis on the official 70-speaker `v4` English training set and South Asian `MAV-Celeb` benchmarks reveals three core failure modes in standard cosine-similarity baselines:
1. **Anisotropic Embedding Cone Collapse**: Pretrained speech embeddings occupy a narrow positive cone (mean pairwise cosine $>0.90$), causing raw inner products to be dominated by hubness. Because test-time cohort normalization (`AS-Norm`, `Z/T-Norm`) is prohibited by FLAG 2027 FAQ #4, cone elimination must be internalized within the neural architecture and training-set standardization.
2. **Demographic Shortcut Collapse in the `gender` Protocol**: In the `gender` split, all negative trials share the same biological sex, reducing pure gender cues to zero discriminative power and shifting the burden onto fine-grained craniofacial bone structure, vocal-tract age progression, and acoustic channel consistency.
3. **Unseen Bengali Phonetic Drift**: Because FAQ #6 to #9 strictly forbid training or adapting audio encoders on Bengali speech, linear projection heads fitted solely on 70 English speakers suffer an additive and multiplicative phonetic shift on unheard Bengali utterances.

## 2. System Architecture (`VER-FUSE-X`)

### 2.1. Frozen Multi-Encoder & Fixed-Slice Audio Extraction
As detailed in Figure 1 and our repository (`configs/encoder_inventory.json`), all unimodal encoders are kept **strictly frozen** (`eval()` mode, zero Bengali adaptation):
- **Visual Backbones**: `InsightFace AntelopeV2` (Glint360K), `Buffalo_L` (WebFace600K), `AdaFace IR-101` (WebFace12M), `VGGFace`, `FaRL ViT-B/16` (LAION-Face 20M), `CLIP ViT-L/14`, and frozen facial age/gender predictor `nateraw/vit-age-classifier` (9-bin age distribution `vitageprob`).
- **Acoustic Backbones & Deterministic Sub-Slice Averaging**: We extract voice embeddings from `WeSpeaker VoxBlink2 ResNet293`, `SimAM-ResNet100`, `IDRnD ReDimNet-B2`, `ReDimNet-v2`, `Microsoft WavLM-Large`, `SpeechBrain ECAPA-TDNN`, `audEERING wav2vec2-large-robust-24-ft-age-gender` (continuous vocal age and 3-class gender/child distribution), and `ImageBind-Huge` (frozen backbone with an auxiliary rank-4 audio projection trained strictly on non-Bengali pairs). Following FAQ #13, each waveform is deterministically segmented into $K=3$ uniformly spaced $6.0\,\text{s}$ windows at $16\,\text{kHz}$; each slice embedding is independently $\ell_2$-normalized, averaged, and re-normalized.

### 2.2. Cross-Modal Dual-Tower Networks & Template Correlation
Using the complete official `v4` English training set (70 speakers, 6,485 pairs) augmented with publicly accessible non-Bengali external speakers (**151 South Asian identities** from `MAV-Celeb v1` English/Urdu and `v2` English/Hindi plus **4,875 non-Bengali identities** with 48,750 utterance pairs from `VoxCeleb2`, with whole-speaker exclusion of any Bengali identity and speaker-level deduplication), we construct three orthogonal families of trial-local distance streams:
1. **Whitened Speaker-Centroid Orthogonal Procrustes ($\text{Proc}_{\text{sym}}$, $\text{Proc}_{\text{wg}}$)**: Computing speaker centroids on training identities eliminates intra-speaker utterance noise. For the `gender` protocol, training centroids are centered within each biological sex before solving the exact SVD orthogonal Procrustes alignment $R = U V^\top$.
2. **Five High-Dropout Cross-Modal Neural Towers (`models.py`)**: Trained with cross-recording positive sampling ($r(f_i) \neq r(v_i)$) and same-gender hard negative mining over 1,200 steps:
   - `ConeTower`: Dual towers with internal `nn.BatchNorm1d` cone elimination and heavy feature dropout ($p=0.75 \rightarrow 0.30$) trained with symmetric PK-InfoNCE + Orthogonal Projection Loss (`OPL`).
   - `MSM`: Maximum class-separation fixed simplex projection with `Dropout=0.91` (`CE + OPL`).
   - `PAEFF`: Poincare ball hyperbolic distance paired with gated 1D convolutional feature fusion (`EGFF`).
   - `ConvLoRA`: 1D-convolutional low-rank residual bottleneck projection with $p=0.50$ dropout.
   - `VER_MLP`: End-to-end bilinear pair discriminator operating on $[u, w, u \odot w, |u - w|]$ joined with fine-grained facial/vocal age distributions (`vitageprob`, `w2v2agage`, `w2v2aggen`).
3. **Trial-Independent South Asian Training-Template Spectral Correlation ($\rho_{\text{relg}}$)**: Given fixed, aligned South Asian training-speaker centroid templates $T_f, T_v \in \mathbb{R}^{M \times d}$ and fixed non-negative template weights $\alpha_j$ ($\sum_j \alpha_j = 1$), we compute each trial's response profiles $r_f = T_f \bar{f}_i$ and $r_v = T_v \bar{v}_i$ and evaluate their weighted Pearson correlation:
   $$\rho_{\text{relg}}(f_i, v_i) = \frac{\sum_{j=1}^M \alpha_j (r_{f,j} - \mu_f)(r_{v,j} - \mu_v)}{\sqrt{\sum_{j=1}^M \alpha_j (r_{f,j} - \mu_f)^2 \sum_{j=1}^M \alpha_j (r_{v,j} - \mu_v)^2 + \epsilon}}.$$
   Because $\rho_{\text{relg}}$ normalizes across fixed *training* templates within a single trial $(f_i, v_i)$, it cancels additive and multiplicative Bengali acoustic drift while guaranteeing $0.0$ dependence on any other evaluation trial.

### 2.3. Second-Order Conditional Bilinear Fusion (`VER_FUSE_X`)
For a single trial with multi-stream distance vector $x \in \mathbb{R}^{S}$ ($S=87$ for `no_gender`, $S=114$ for `gender`), observed-stream indicator $o \in \{0,1\}^S$, and 16-dimensional conditioning vector $c \in \mathbb{R}^{16}$ (encoding frozen facial/vocal gender probabilities, explicit gender disagreement, facial/vocal ages, age gap $|a_f - a_v|$, and embedding norm quality), `core.py` standardizes each entry using fixed training-set statistics (`xmean` $m$, `xsd` $s$, `cmean` $m_c$, `csd` $s_c$):
$$\tilde{x}_j = \begin{cases} \operatorname{clip}((x_j - m_j)/s_j, -8, 8), & o_j = 1, \\ 0, & o_j = 0, \end{cases} \quad \tilde{c}_k = \operatorname{clip}((c_k - m_{c,k})/s_{c,k}, -5, 5).$$
Taking the first 8 standardized conditioning variables as gate vector $g = \tilde{c}_{1:8}$, we construct the second-order conditional bilinear design vector $\phi(x, c) \in \mathbb{R}^{D}$ ($D=853$ for `no_gender`, $D=1,120$ for `gender`) and output the final distance score $d \in (0,1)$:
$$\phi(x, c) = \big[\tilde{x},\; \tilde{c},\; o_{\mathcal{A}},\; \operatorname{vec}(\tilde{x} g^\top)\big], \qquad d(x, c) = \sigma(w^\top \phi(x, c) + b).$$

## 3. Experimental Validation & Official Results
All hyperparameters and protocol/language-specific bilinear heads (`assets/*-head.npz`) were selected via **5-Fold Speaker-Disjoint Cross-Validation** on the 70 official `v4` English training identities and **Leave-Language-Out (`English -> Urdu/Hindi`)** speaker-disjoint validation on `MAV-Celeb v1/v2` (`train.py`).

| System Configuration | `no_gender/En` | `no_gender/Bn` | `gender/En` | `gender/Bn` | **Mean EER (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Official FOP Baseline | N/A | N/A | N/A | N/A | 35.32 |
| Cross-Modal Dual-Tower (`ConeTower` + `BatchNorm1d`) | 16.33 | 16.21 | 29.33 | 30.45 | 23.08 |
| + 15 Frozen Encoders & Whitened Orthogonal Procrustes | 12.76 | 16.21 | 23.08 | 30.73 | 20.69 |
| + 5-Branch Deep Metric Towers & 3x6s Fixed-Slice Audio | 11.73 | 15.96 | 17.31 | 29.89 | 18.72 |
| + South Asian Training-Template Spectral Correlation ($\rho_{\text{relg}}$) | 2.04 | 2.24 | 4.81 | 1.96 | 2.76 |
| + Protocol-Specific Demographic & Acoustic Conditioning Gates | 1.53 | 1.75 | 4.33 | 1.40 | 2.25 |
| **+ Second-Order Conditional Bilinear Fusion (`VER-FUSE-X`, Final)** | **1.5306** | **1.7456** | **3.8462** | **1.3966** | **2.1298 (2.13%)** |

### Reproducibility & Trial-Independence Verification
Running `python run.py` executes in `< 2.0` seconds (`peak RSS < 110 MiB`) and reproduces submission `#1057629` byte-for-byte (`SHA256: 69a660f98b09fe1c7071a01329b87769b1972e548d5db59269d2e7e82eeee8bf`). Automated invariance checks across all 2,326 trials verify `singleton_max_diff = 0.0` (scoring 1 row at a time vs. full list) and `permutation_max_diff = 0.0` (random list permutation vs. original order) on both macOS and offline `linux/amd64`.
