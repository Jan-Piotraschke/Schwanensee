# Physio-SensAI

*Sharing the tacit knowledge of experienced workers*

Creating a PINN softsensor, which adjusts the parameters of its underlying physical model.

What is this about? I want to recite Faust, Goethe for the answer:

<blockquote>
Der Erdenkreis ist mir genug bekannt.<br>
Nach drüben ist die Aussicht uns verrannt;<br>
Tor, wer dorthin die Augen blinzelnd richtet,<br>
Sich über Wolken seinesgleichen dichtet!
</blockquote>

This tension illustrates that a complete description of tissue or other physiological diagnostics may not be possible with our current methods, and therefore, we may resort to a hyperspace approach of the neural network, since its understanding of the system could differ from ours, which, however, could prove beneficial to us.

## Usage

```bash
pip install deepxde tensorflow-probability tf-keras
```

```bash
python src/psai.py --python --input example/lorenz/generated/Lorenz_1963.py
```

and then after this

```bash
DDE_BACKEND=tensorflow.compat.v1 python example/lorenz/lorenz_w_exogenous_stimulus.py
```

### ONNX Converter

```bash
python -m tf2onnx.convert --saved-model generalized_patient --output model.onnx
```

## Code Format

```bash
ruff format src
ruff check src
```

## Briefer about PINNs

[PINNs](https://maziarraissi.github.io/PINNs/) can be designed to solve two classes of problems:
- data-driven solution (forward problem)
- data-driven discovery (inverse problem)

of differential equations e.g. partical differential equations (PDE).

Here we implemented the **data-driven discovery** given noisy and incomplete measurements.
It is important to understand that the PDEs (that govern a given data-set), or in generell the xDEs, get embeded into the learning process of the NN.
Explicitly speaking, the PDEs get embeded into the cost function of the NN. This is done using the DeepXDE package.
With that, the embeded PDEs act as a regularization agent that limits the space of admissible solutions of the NN training.
The PINN alone does not find any unknown/missing terms of the PDE problem.
**It only adjusts the unknown PDE parameters** as part of its cost function.
