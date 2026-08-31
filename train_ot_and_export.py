import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

# Hyperparameters aligned with DSP Micro-Pipeline
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160  # 10ms frame step
WIN_LENGTH = 400  # 25ms window length
N_MFCC = 13
N_MELS = 32
TARGET_FRAMES = 97  # (16000 - 400)/160 + 1 = 98 ~ padded/trimmed to 97 frames
MODEL_INPUT_SIZE = N_MFCC * TARGET_FRAMES  # 1261
CLASSES = ["up", "down", "left", "right", "back", "go"]


class SpeechDataset(SPEECHCOMMANDS):
    def __init__(self, subset: str = 'training'):
        super().__init__('./data', download=True, subset=subset)
        self.labels = CLASSES
        self.indices = []
        for idx in range(len(self._walker)):
            rel_path = self._walker[idx]
            label = os.path.basename(os.path.dirname(rel_path))
            if label in self.labels:
                self.indices.append(idx)

    def __getitem__(self, n: int):
        actual_idx = self.indices[n]
        waveform, sample_rate, label, _, _ = super().__getitem__(actual_idx)

        # Standardize audio to 1 second (16000 samples)
        if waveform.shape[1] < SAMPLE_RATE:
            waveform = nn.functional.pad(waveform, (0, SAMPLE_RATE - waveform.shape[1]))
        else:
            waveform = waveform[:, :SAMPLE_RATE]

        # MFCC Pipeline (Matching Embedded C DSP setup)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={
                'n_fft': N_FFT,
                'win_length': WIN_LENGTH,
                'hop_length': HOP_LENGTH,
                'n_mels': N_MELS,
                'center': False
            }
        )
        mfcc = mfcc_transform(waveform)  # Shape: [1, 13, 97]
        label_idx = self.labels.index(label)
        
        # Flatten input explicitly to [1261] for clean static C code generation
        flat_mfcc = mfcc.squeeze(0).view(-1)
        return flat_mfcc, label_idx

    def __len__(self):
        return len(self.indices)


class EmbeddedKWSModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Flattened Input: 13 MFCCs x 97 frames = 1261 features
        self.fc1 = nn.Linear(MODEL_INPUT_SIZE, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input expected as [batch_size, 1261]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def post_process_generated_header(header_path: str):
    """
    Injects extern 'C' wrappers, memory alignment macros, and dynamic array 
    attributes directly into the generated C header file.
    """
    if not os.path.exists(header_path):
        print(f"Header file {header_path} not found. Skipping post-processing.")
        return

    with open(header_path, 'r') as f:
        content = f.read()

    # Define alignment macro if missing
    alignment_macro = (
        "\n#if defined(__GNUC__) || defined(__clang__)\n"
        "  #define KWS_ALIGN32 __attribute__((aligned(32)))\n"
        "#else\n"
        "  #define KWS_ALIGN32\n"
        "#endif\n"
    )

    # Insert C++ extern guard
    cpp_guard_start = "\n#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
    cpp_guard_end = "\n#ifdef __cplusplus\n}\n#endif\n"

    # Inject macros right after includes
    if "#include" in content:
        last_inc = content.rfind("#include")
        inc_end = content.find("\n", last_inc)
        content = content[:inc_end+1] + alignment_macro + cpp_guard_start + content[inc_end+1:]
    else:
        content = alignment_macro + cpp_guard_start + content

    # Add KWS_ALIGN32 to float array declarations
    content = re.sub(r'(extern\s+float\s+\w+\[)', r'extern KWS_ALIGN32 float \1', content)

    # Wrap before trailing endif
    if "#endif" in content:
        last_endif = content.rfind("#endif")
        content = content[:last_endif] + cpp_guard_end + content[last_endif:]

    with open(header_path, 'w') as f:
        f.write(content)

    print(f"Post-processed {header_path} with alignment macros and extern 'C' guards.")


def main():
    os.makedirs('./data', exist_ok=True)
    dataset = SpeechDataset(subset='training')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddedKWSModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training initial model on Google Speech Commands...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, MODEL_INPUT_SIZE).to(device)
    
    onnx_path = "kws_model.onnx"
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["audio_features"],
        output_names=["kws_logits"]
    )
    print(f"ONNX Model successfully saved to {onnx_path}")

    # Example transpiler call (Uncomment if invoking transpiler directly from script):
    # os.system(f"emx-onnx-cgen {onnx_path} -o generated_kws_model")
    
    # Run post-processing on the target generated header
    post_process_generated_header("generated_kws_model/kws_model.h")


if __name__ == "__main__":
    main()