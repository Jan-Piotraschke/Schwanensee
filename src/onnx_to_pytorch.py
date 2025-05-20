import onnx
import torch
from onnx2torch import convert
import numpy as np
import onnxruntime as ort

# Load ONNX model
onnx_model_path = "model.onnx"
onnx_model = onnx.load(onnx_model_path)
torch_model_2 = convert(onnx_model)

# Get expected input shape from ONNX model
input_tensor = onnx_model.graph.input[0]
input_name = input_tensor.name
input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
print("Expected input shape:", input_shape)

# Generate input with matching shape
x = torch.ones((1, 1))  # batch of 1, 1 feature

# Forward through PyTorch model
out_torch = torch_model_2(x)

# Inference with ONNX Runtime
ort_sess = ort.InferenceSession(onnx_model_path)
outputs_ort = ort_sess.run(None, {input_name: x.numpy()})
outputs_ort_np = outputs_ort[0]

# Compare outputs
print(torch.max(torch.abs(torch.tensor(outputs_ort_np) - out_torch.detach())))
print(np.allclose(outputs_ort_np, out_torch.detach().numpy(), atol=1.0e-7))

# Save the converted PyTorch model
torch.save(torch_model_2.state_dict(), "converted_model.pth")
torch.save(torch_model_2, "converted_model_full.pt")
