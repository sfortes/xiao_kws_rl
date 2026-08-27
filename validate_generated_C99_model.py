import onnx
import numpy as np
from cffi import FFI
import os # Import os to set environment variables

# Load the ONNX model to extract weights for Python validation
model = onnx.load("kws_model.onnx")
graph = model.graph

weights_dict = {}
for initializer in graph.initializer:
    weights_dict[initializer.name] = onnx.numpy_helper.to_array(initializer)

# Get weights and biases by their names
fc1_w_np = weights_dict['fc1.weight'] # Shape: [64, 1261]
fc1_b_np = weights_dict['fc1.bias']   # Shape: [64]
fc2_w_np = weights_dict['fc2.weight'] # Shape: [6, 64]
fcn2_b_np = weights_dict['fc2.bias'] # Shape: [6]

# Generate a dummy input (flattened MFCC like the model expects)
dummy_input_np = np.random.randn(1, 13 * 97).astype(np.float32)

# --- Python Forward Pass (for validation) ---
def python_inference(input_data, fc1_w, fc1_b, fc2_w, fc2_b):
    # Flatten input if not already
    input_data = input_data.flatten()

    # FC1 + ReLU
    hidden = np.dot(input_data, fc1_w.T) + fc1_b
    hidden[hidden < 0] = 0 # ReLU

    # FC2
    output = np.dot(hidden, fc2_w.T) + fc2_b
    return output

python_output = python_inference(dummy_input_np, fc1_w_np, fc1_b_np, fc2_w_np, fcn2_b_np)

print("Python Inference Output (first 5 values):", python_output[:5])

# --- C99 Forward Pass ---
ffi = FFI()

# Define constants for array sizing in Python (derived from C header)
MODEL_INPUT_SIZE = 1261
MODEL_HIDDEN_SIZE = 64
MODEL_OUTPUT_SIZE = 6

# Define the C structures and functions from the header
# MODIFIED: Removed all #define and extern declarations from ffi.cdef.
# Only the function signature for model_inference is kept.
ffi.cdef("""
    void model_inference(const float* input, float* output);
""")

# Set CFFI_DEBUG environment variable for verbose output
os.environ['CFFI_DEBUG'] = '1'

# Compile the C code
# The current directory is where kws_model.c and kws_model.h are located
lib = ffi.verify(sources=['kws_model.c'], include_dirs=['.'], extra_compile_args=['-O3', '-lm'])

# Prepare input and output buffers for C
c_input = ffi.cast("float*", dummy_input_np.ctypes.data)
c_output = ffi.new("float[]", MODEL_OUTPUT_SIZE)

# Call the C function
lib.model_inference(c_input, c_output)

# Convert C output to numpy array
c_output_np = np.array([c_output[i] for i in range(MODEL_OUTPUT_SIZE)])

print("C99 Inference Output (first 5 values):", c_output_np[:5])

# --- Compare Outputs ---
# Both python_output and c_output_np are 1D arrays of shape (6,)
comparison = np.isclose(python_output, c_output_np, rtol=1e-5, atol=1e-6)

if np.all(comparison):
    print("\nValidation Successful: Python and C99 outputs match!")
else:
    print("\nValidation Failed: Mismatch found between Python and C99 outputs.")
    print("Differences:")
    print(python_output - c_output_np)
