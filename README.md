# Physio-SensAI

*Sharing the tacit knowledge of experienced workers*

Creating a PINN softsensor, which adjusts the parameters of its underlying physical model.

## Usage

```bash
pip install deepxde tensorflow-probability tf-keras
```

```bash
DDE_BACKEND=tensorflow python src/model/lorenz_w_exogenous_stimulus.py
DDE_BACKEND=tensorflow python src/floquet_mode.py
DDE_BACKEND=tensorflow python src/heat.py
```

## Code Format

```bash
ruff format src
ruff check src
```
