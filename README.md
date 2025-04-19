# 🧠 Rumor Propagation with Modular Fourier Neural Operators

This repo simulates and learns the dynamics of **rumor spreading** on networks using **Fourier Neural Operators (FNOs)**.

The system is modular, interpretable, and frequency-aware — enabling both forward modeling and inverse parameter inference, with interpretable insights into learned frequency modes.

---

## 📦 Features

- ✅ Simulate rumor dynamics on ER, BA, WS graphs
- ✅ Train **FNOs** to learn time evolution of SIR systems
- 🔁 Invert trajectories → model parameters with inverse FNOs
- 📈 Bifurcation surface generation over α, β, δ parameter sweeps
- 🧠 Visualize learned **spectral filters** and frequency preferences
- 🔎 Explore effect of network topology on rumor behavior

---

## 🧬 Models Included

Each model implements:

```python
.simulate(params, T, dt, ...)
.parameter_dim()
.state_dim()
```

### 🔹 `DongRumorModel`  
ODE-based SIR-type model from Dong et al. (2018)

### 🔹 `SIRModel`  
Standard SIR rumor model on fixed graph topologies

### 🔹 `DegreeAwareSIRModel`  
Agent-based SIR model using degree-dependent transition probabilities \( P(k'|k) \) — now supports FNO-based learning and full parameter inversion

---

## 📂 Project Structure

```
modular_fno/
├── dynamics/               # Simulation models
├── models/                 # FNO + spectral conv layers
├── scripts/                # Training, visualization, spectral analysis
├── utils/                  # Dataset generation, normalization
├── run_pipeline.py         # Unified forward + inverse training CLI
├── data/                   # Saved datasets
├── checkpoints/            # Trained model weights
```

---

## 🔧 Training & Inference

### Forward Training (params → trajectory)
```bash
python run_pipeline.py train_forward --epochs 50
```

### Inverse Training (trajectory → params)
```bash
python run_pipeline.py train_inverse --epochs 50
```

### Full Pipeline
```bash
python run_pipeline.py all --epochs 50
```

---

## 🎨 Spectral Visualizations

Learned spectral filters are interpretable and can be visualized:

### 🔹 Fourier Filter Line Plots
```bash
python scripts/visualize_fourier_filters.py   --checkpoint checkpoints/fno_forward_heterogeneous.pth   --in_channels 5 --out_channels 3
```

### 🔹 Filter Matrix View (per layer, per input/output channel pair)
```bash
python scripts/visualize_filter_matrices.py
```

### 🔹 Aggregated Frequency Spectrum
```bash
python scripts/plot_aggregated_spectrum.py   --checkpoint checkpoints/fno_forward_heterogeneous.pth   --in_channels 5 --out_channels 3
```

This reveals which Fourier modes each FNO layer focuses on — showing how early layers capture global dynamics and later layers refine fine-grain variations.

---

## 💾 Data Format

Saved datasets in `data/` follow this format:
- `.pth` files with `[x_tensor, y_tensor]`
- Forward mode: `x = [params + time]`, `y = trajectory`
- Inverse mode: `x = [trajectory + time]`, `y = params`

---

## 📊 Requirements

```bash
pip install -r requirements.txt
```

- Python ≥ 3.10
- PyTorch ≥ 1.12
- matplotlib, numpy, tqdm, seaborn, networkx

---

## ⚡ Future Extensions

- [ ] Topology-aware inverse learning (graph statistics → trajectory)
- [ ] Transformer-style operator on graph signals
- [ ] Live interactive bifurcation sliders
- [ ] Compare learned filters across graph types (ER vs BA vs WS)

---

## 👨‍🔬 Author

**Jacob Briones**  
Applied mathematician, neural operator whisperer, and full-time dad. Essentially self taught.
