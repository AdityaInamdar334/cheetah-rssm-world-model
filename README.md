# Dreamer-style Recurrent State Space Model (RSSM) for Cheetah Run

This repository contains a self-contained, high-performance implementation of a **Dreamer-style World Model (Recurrent State Space Model or RSSM)** trained on pixel observations ($64 \times 64$ RGB) from the DeepMind Control Suite `cheetah run` environment. 

The pipeline is optimized for macOS, featuring **Apple Silicon GPU acceleration (MPS)**, achieving a **27.4x speedup** compared to standard CPU training.

---

## 🚀 Key Highlights & Results

- **Performance Acceleration**: Optimized for macOS GPU execution using PyTorch's Metal Performance Shaders (MPS) backend.
  - **CPU Training Speed**: `0.18 steps/s` (~5.68 seconds/step) ➔ **ETA: 6.4 days**
  - **MPS GPU Training Speed**: `4.94 steps/s` (~0.20 seconds/step) ➔ **ETA: 5.6 hours (27.4x faster!)**
- **High-Fidelity Reconstructions**: Achieved a visual reconstruction MSE of **`0.000640`** (closed-loop).
- **Stable Open-Loop Dreaming**: Evaluated the RSSM cell's stability during long-horizon open-loop imagination rollouts (30 steps), achieving a prediction MSE of **`0.000670`** without diverging.
- **Interactive Web Showcase**: A clean, self-contained single-page dashboard built using HTML/CSS/JS, featuring an interactive frame-by-frame dreaming simulator slider and diagnostic plots.

---

## 📂 Repository Structure

- `requirements.txt`: Python package configurations with pinned, verified macOS-compatible versions of MuJoCo (`3.8.1`) and dm_control (`1.0.41`).
- `collect_data.py`: Creates the pixel observation cheetah environment and uses a random policy to gather a 20,000-step exploration dataset (`cheetah_run_dataset.pkl`).
- `utils.py`: Contains the `EpisodeReplayBuffer` which segments raw transitions into discrete episodes of 1,000 steps and handles batch sequence preprocessing (CHW transposition & normalization).
- `world_model.py`: Implements the neural networks for the world model:
  - **Encoder**: CNN mapping $64 \times 64 \times 3$ image frames to $1024$-dim embedding space.
  - **RSSM Cell**: A Recurrent State Space Model cell utilizing a GRU deterministic state ($200$-dim) and a Gaussian stochastic state ($30$-dim) with prior/posterior distribution outputs.
  - **Decoder**: Transpose CNN mapping latent states back to reconstructed pixels.
  - **Reward Predictor**: MLP predicting scalar rewards from latent states.
- `train.py`: The main training script supporting CPU/GPU device selection, gradient norm clipping, and automatic checkpointing.
- `visualize_predictions.py`: Generates the side-by-side animated comparison GIF between ground truth and predicted frames.
- `visualize_diagnostics.py`: Compiles validation MSE error curves and reward prediction accuracy charts.
- `export_assets_for_web.py`: Exports individual step frames as PNGs for the interactive web slider.
- `web/`: The self-contained static website folder containing:
  - `index.html`: Interactive web dashboard.
  - `diagnostics.png`: Validation error and reward curves.
  - `prediction_comparison.gif`: Animated rollout compilation.
  - `assets/`: Folder housing the individual step frames for the dreaming slider.

---

## 🛠️ Installation & Setup

We recommend using **Python 3.11** to ensure pre-compiled wheels are fetched automatically, bypassing compiler errors for `labmaze`/`dm_control` on macOS.

1. **Create and activate a virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Install all dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Execution Workflow

### Step 1: Collect Exploration Data
Gather random policy exploration transitions ($64 \times 64$ RGB observations):
```bash
python collect_data.py
```
*Outputs: `cheetah_run_dataset.pkl` (236.8 MB)*

### Step 2: Train the World Model
Train the RSSM for 100,000 steps. The script will automatically detect and prioritize your Apple Silicon GPU (`mps`):
```bash
python train.py
```
*Outputs: checkpoints saved to `checkpoints/` and final weights saved to `final_model.pt`*

### Step 3: Run Evaluation & Diagnostics
Evaluate the model's visual reconstruction and reward predicting accuracy:
```bash
# Generate the animated prediction comparison GIF
python visualize_predictions.py

# Generate the MSE and Reward validation curve plots
python visualize_diagnostics.py
```
*Outputs: `prediction_comparison.gif` and `diagnostics.png`*

### Step 4: Export Web Assets & Launch Dashboard
Prepare the slider assets and run the interactive web dashboard locally:
```bash
# Export the individual frames
python export_assets_for_web.py

# Copy the charts and animations to the web folder
cp diagnostics.png web/
cp prediction_comparison.gif web/

# Launch the local server
python3 -m http.server 8000 --directory web
```
Navigate to **[http://localhost:8000](http://localhost:8000)** in your browser to view the interactive player!

---

## 🧠 RSSM Architecture Detail

The Recurrent State Space Model splits transition dynamics into a **deterministic component** (recurrent state $h_t$) to memorize past events, and a **stochastic component** (latent state $s_t$) to model uncertainty.

```
                  [Observation x_t]
                          │
                   (CNN Encoder)
                          │
                          ▼
                    [Embedding e_t]
                          │
                          ▼
 [prev State s_t-1] ──► (GRU Cell) ──► [determ State h_t] ──► (Decoder) ──► [Reconstructed x_t]
                          │                     │
                          ▼                     ▼
                  (Posterior q) ◄─────── (Prior p)
                          │                     │
                          ▼                     ▼
                  [stoch State s_t]     [predicted s_t]
```

At every time step, the model computes:
1. **Deterministic transition**: $h_t = \text{GRU}(h_{t-1}, s_{t-1}, a_{t-1})$
2. **Prior distribution**: $p(s_t \mid h_t)$
3. **Posterior distribution**: $q(s_t \mid h_t, e_t)$ where $e_t$ is the CNN embedding of $x_t$.
4. **Loss**: Reconstruction MSE + Reward MSE + $\text{KL}(q(s_t) \parallel p(s_t))$ with a $1.0$ free bits constraint to prevent latent collapse.
