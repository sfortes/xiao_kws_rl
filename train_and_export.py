# !pip install onnxscript

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

# Hyperparameters aligned with DSP Micro-Pipeline
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160 # 10ms frame step
WIN_LENGTH = 400 # 25ms window length
N_MFCC = 13
N_MELS = 32
TARGET_FRAMES = 97 # (16000 - 400)/160 + 1 = 98 ~ padded/trimmed to 97 frames
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
        mfcc = mfcc_transform(waveform) # Shape: [1, 13, 97]
        label_idx = self.labels.index(label)
        return mfcc.squeeze(0), label_idx

    def __len__(self):
        return len(self.indices)

class EmbeddedKWSModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Flattened Input: 13 MFCCs x 97 frames = 1261 features
        self.fc1 = nn.Linear(13 * 97, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    os.makedirs('./data', exist_ok=True) # Add this line to create the directory
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
    dummy_input = torch.randn(1, 13, 97).to(device)
    torch.onnx.export(
        model, dummy_input, "kws_model.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes=None, opset_version=11
    )
    print("ONNX Model saved to kws_model.onnx")

if __name__ == "__main__":
    main()
