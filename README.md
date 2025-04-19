# 🧠 Rumor Propagation with Modular Fourier Neural Operators

A fully modular, PyTorch-powered framework for simulating, learning, and inferring **rumor dynamics** over networks using **Fourier Neural Operators (FNOs)**. Built to explore not just how information spreads—but what it reveals about the network itself.

## 🚀 What This Repo Supports

- ✅ Forward simulations of rumor spread (Dong, SIR, Topo-based models)
- 🔁 Inverse learning to recover model parameters from observed dynamics
- 📈 Bifurcation and sensitivity analysis with precomputed heatmaps
- 🧠 Inference of network **topological features** (clustering, path length) from observed trajectories
- 📊 Clean CLI + script interfaces for all training and evaluation routines
- 🔌 Drop-in modular support for defining new models

---

## 📂 Repository Structure

```
rumor_spread/
├── dynamics/               # All pluggable simulation models
│   ├── base.py            # Abstract model interface
│   ├── dong_model.py      # Discrete-time Dong rumor model
│   ├── sir_model.py       # SIR epidemic model (partially implemented)
│   └── topo_model.py      # Rumor spreading with topological descriptors
│
├── models/                # Fourier Neural Operator implementations
│   ├── fno.py             # FNO1d model definition
│   └── spectral_conv.py   # 1D spectral convolution building block
│
├── utils/                 # Utility code for dataset generation
│   └── data_generation.py # Generic time-aware dataset builder
│
├── scripts/               # CLI-ready training & visualization scripts
│   ├── train_forward.py
│   ├── train_inverse.py
│   ├── train_topology_inverse.py
│   ├── visualize_inverse_topology.py
│   └── test_fno_topology.py
│
├── run_pipeline.py        # Full CLI pipeline controller
├── inference.py           # Inference + visual output
├── figures/               # Auto-generated figures, heatmaps
├── checkpoints/           # Trained models
└── plots/                 # Visual output (e.g., inverse topology scatter)
```

---

## 🧬 Simulation Models (All Subclass `DynamicalSystem`)

Each model must define:
```python
.simulate(params, T, dt)
.parameter_dim()
.state_dim()
```

Current implementations:
- **DongRumorModel** – rumor with forgetting and saturation
- **SIRModel** – over graph topology
- **TopoRumorModel** – rumor + dynamic topology + descriptor prediction (clustering, path length, assortativity)

---

## 🧠 Learning Tasks

### ➤ Forward Problem
**Goal:** Learn \( u(t) \) given parameters \( \theta = (\beta, \alpha, \delta, i_0) \)
```bash
python run_pipeline.py train_forward --epochs 100
```

### ➤ Inverse Problem
**Goal:** Recover \( \theta \) given trajectory \( u(t) \)
```bash
python run_pipeline.py train_inverse --epochs 100
```

### ➤ Topology Inference (Experimental)
**Goal:** Predict clustering, path length, assortativity from observed rumor dynamics
```bash
python scripts/train_topology_inverse.py
```

**Visualization:**
```bash
python scripts/visualize_topology_inverse.py --samples 30
```

---

## 📊 Requirements

```bash
pip install -r requirements.txt
```
- Python ≥ 3.10
- PyTorch ≥ 1.12
- matplotlib, numpy, tqdm, seaborn, networkx

---

## 🤔 Who Should Use This?

- ML researchers exploring neural operators
- Network scientists modeling information diffusion
- Physicists studying complex systems
- Anyone trying to infer graph properties from observed dynamics

---

## 📜 License

MIT — use it, fork it, cite it, build weird stuff on top of it.

---

## 👨‍💻 Author

**Jacob Briones**  
*Sexy Nerd. Father. Dynamical system maximalist. Philosopher?*

