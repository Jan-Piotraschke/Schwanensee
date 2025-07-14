import onnx
import torch
import numpy as np
import onnxruntime as ort
from onnx2torch import convert

# Load ONNX model
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)
torch_model_2.eval()  # Set to evaluation mode

# Get expected input shape from ONNX model
input_tensor = onnx_model.graph.input[0]
input_name = input_tensor.name

# Fix: Replace 0 dimensions (often used as symbolic batch sizes) with 1
input_shape = [
    dim.dim_value if dim.dim_value > 0 else 1
    for dim in input_tensor.type.tensor_type.shape.dim
]
print("Fixed input shape for dummy input:", input_shape)

# Create dummy input with the corrected shape
x = torch.ones(*input_shape)

# Forward through PyTorch model
out_torch = torch_model_2(x)

# Inference with ONNX Runtime for comparison
ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {input_name: x.numpy()})
outputs_ort_np = outputs_ort[0]

# Compare outputs
max_diff = torch.max(torch.abs(torch.tensor(outputs_ort_np) - out_torch.detach()))
print("Max output difference:", max_diff.item())
print(
    "Outputs match (within tolerance)?",
    np.allclose(outputs_ort_np, out_torch.detach().numpy(), atol=1.0e-7),
)

# Save the raw model (not for C++ usage – just for Python reference/debug)
torch.save(torch_model_2.state_dict(), "converted_model.pth")
torch.save(torch_model_2, "converted_model_full.pt")

# Convert to TorchScript (for use in C++)
traced_model = torch.jit.trace(torch_model_2, x)
traced_model.save("traced_model.pt")

print("TorchScript model saved as traced_model.pt (C++ compatible)")
