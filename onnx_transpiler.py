import onnx
import numpy as np

def generate_c99_model(onnx_path, output_header, output_source):
    model = onnx.load(onnx_path)
    graph = model.graph

    weights = {}
    for initializer in graph.initializer:
        weights[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # Resolve Weight Names by explicit name
    fc1_w = weights['fc1.weight'] # Shape: [64, 1261]
    fc1_b = weights['fc1.bias'] # Shape: [64]
    fc2_w = weights['fc2.weight'] # Shape: [6, 64]
    fc2_b = weights['fc2.bias'] # Shape: [6]

    with open(output_header, "w") as f:
        f.write("#ifndef KWS_MODEL_H\n#define KWS_MODEL_H\n\n#include <stdint.h>\n\n")
        f.write("#define MODEL_INPUT_SIZE 1261\n")
        f.write("#define MODEL_HIDDEN_SIZE 64\n")
        f.write("#define MODEL_OUTPUT_SIZE 6\n\n")
        f.write("// Mutable dynamic weights residing in SRAM for Core 1 RL training updates\n")
        f.write("extern float fc1_w[MODEL_HIDDEN_SIZE][MODEL_INPUT_SIZE];\n")
        f.write("extern float fc1_b[MODEL_HIDDEN_SIZE];\n")
        f.write("extern float fc2_w[MODEL_OUTPUT_SIZE][MODEL_HIDDEN_SIZE];\n")
        f.write("extern float fc2_b[MODEL_OUTPUT_SIZE];\n\n")
        f.write("void model_inference(const float* input, float* output);\n")
        f.write("#endif\n")

    with open(output_source, "w") as f:
        f.write('#include "kws_model.h"\n#include <math.h>\n\n')

        # Write arrays to C code
        f.write("float fc1_w[MODEL_HIDDEN_SIZE][MODEL_INPUT_SIZE] = {\n")
        for row in fc1_w:
            f.write("  {" + ", ".join(f"{v:.7f}f" for v in row) + "},\n")
        f.write("};\n\n")

        f.write("float fc1_b[MODEL_HIDDEN_SIZE] = {\n  ")
        f.write(", ".join(f"{v:.7f}f" for v in fc1_b))
        f.write("\n};\n\n")

        f.write("float fc2_w[MODEL_OUTPUT_SIZE][MODEL_HIDDEN_SIZE] = {\n")
        for row in fc2_w:
            f.write("  {" + ", ".join(f"{v:.7f}f" for v in row) + "},\n")
        f.write("};\n\n")

        f.write("float fc2_b[MODEL_OUTPUT_SIZE] = {\n  ")
        f.write(", ".join(f"{v:.7f}f" for v in fc2_b))
        f.write("\n};\n\n")

        # C99 Forward Pass Function
        f.write("""void model_inference(const float* input, float* output) {
    float hidden[MODEL_HIDDEN_SIZE];

    // Layer 1: FC + ReLU
    for (int i = 0; i < MODEL_HIDDEN_SIZE; i++) {
        float sum = fc1_b[i];
        for (int j = 0; j < MODEL_INPUT_SIZE; j++) {
            sum += input[j] * fc1_w[i][j];
        }
        hidden[i] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    // Layer 2: FC (Logits)
    for (int i = 0; i < MODEL_OUTPUT_SIZE; i++) {
        float sum = fc2_b[i];
        for (int j = 0; j < MODEL_HIDDEN_SIZE; j++) {
            sum += hidden[j] * fc2_w[i][j];
        }
        output[i] = sum;
    }
}
""")
    print(f"Transpiled C99 sources generated: {output_header}, {output_source}")

if __name__ == "__main__":
    generate_c99_model("kws_model.onnx", "kws_model.h", "kws_model.c")