# xiao_kws_rl
Pure C99 language Deep Learning inference firmware voice detection with reinforcement learning on esp32

 Operation:

The deployed architecture operates as a **closed-loop edge learning system** directly on the Seeed Studio XIAO ESP32-S3 Sense.
It continuously captures real-world microphone input, converts raw acoustics into deep feature representations, runs low-latency inference on Core 0, and uses environmental feedback to perform online reinforcement learning updates directly in internal SRAM on Core 1—all without taking down or pausing the real-time audio pipeline.

System Architecture:

[ HOST PC ]
1. PyTorch Dataset & Model  --> Trains 1D-CNN on 6 target words using DataLoader
2. ONNX Export             --> Serializes trained model into static graph (.onnx)
3. ONNX to C Transpilation  --> Converts graph + model parameters to zero-dependency C99

 [ SEEED STUDIO XIAO ESP32-S3 SENSE ]
 - Shared SRAM Memory (DRAM)
 - Global Weights Buffer (g_model_weights)  <-- Dynamic weights, fine-tunable at runtime
 - FreeRTOS Mutex (g_weights_mutex)       <-- Prevents concurrent read/write collisions
 - FreeRTOS Queue (g_inference_queue)     <-- Reserved for the future Core 0 -> Core 1 RL handoff
 
 CORE 0 TASK (Audio Sampling & Model Execution)
 1. Reads raw 16kHz PCM audio stream via PDM hardware peripheral (GPIO 42 CLK, GPIO 41 DAT)
 2. Extracts MFCC features (13 coefficients x 97 time frames = 1261 floats)
 3. Locks Mutex -> Reads SRAM weights -> Executes C99 ONNX forward pass -> Unlocks Mutex
 4. Applies Softmax to produce probabilities for "UP", "DOWN", "LEFT", "RIGHT", "BACK", "GO"
 5. Logs the highest-confidence class; queue publication is not enabled in validation mode
 
 CORE 1 TASK (Online Reinforcement Learning Update)
 1. Waits on the reserved Queue for a future inference payload from Core 0
 2. Evaluates environment feedback / confidence metric to assign Reward (+1.0 or -1.0)
 3. Computes policy gradient update on final classification layer: ΔW = lr * reward * activation
 4. Locks Mutex -> Modifies `g_model_weights` directly in SRAM -> Unlocks Mutex

## Debian Toolchain, Model Generation, and Device Workflow

This project targets the Seeed Studio XIAO ESP32-S3 Sense with its onboard PDM
microphone and 8 MB PSRAM. The firmware target is `esp32s3`. The commands below
assume a Debian or Ubuntu host and are run from the repository root unless a
command says otherwise.

### 1. Install host prerequisites

Install the packages required by ESP-IDF, Git, Python virtual environments, and
the Python packages used to train, export, and inspect the model:

```bash
sudo apt update
sudo apt install -y \
	git wget flex bison gperf cmake ninja-build ccache libffi-dev libssl-dev \
	dfu-util libusb-1.0-0 python3 python3-pip python3-venv
```

Clone this repository and enter it:

```bash
git clone https://github.com/sfortes/xiao_kws_rl.git
cd xiao_kws_rl
```

### 2. Install ESP-IDF v6.0.2

The current project configuration was built and tested with ESP-IDF `v6.0.2`.
Use a separate directory for the framework and tools:

```bash
mkdir -p ~/esp
cd ~/esp
git clone --branch v6.0.2 --depth 1 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3
```

The installer downloads the Xtensa ESP32-S3 compiler, CMake/Ninja support, and
the ESP-IDF Python environment. It does not need to be repeated for every
project clone.

Activate ESP-IDF in every new shell before using `idf.py`:

```bash
source ~/esp/esp-idf/export.sh
```

If ESP-IDF was installed through Espressif Installation Manager instead, use
the activation script for the installed version. This is the setup used for the
baseline validation of this repository:

```bash
source ~/.espressif/tools/activate_idf_v6.0.2.sh
```

Confirm that the correct environment is active:

```bash
idf.py --version
```

The output should identify ESP-IDF `v6.0.2`.

### 3. Set up the Python model environment

Keep model-training dependencies separate from the ESP-IDF Python environment:

```bash
cd ~/xiao_kws_rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio onnx numpy cffi
```

Training downloads the Google Speech Commands dataset into `./data` on its
first run. It can take significant disk space and time. The class order used by
the model and firmware is:

```text
UP, DOWN, LEFT, RIGHT, BACK, GO
```

### 4. Train and export an ONNX model

The current preferred training script is `train_ot_and_export.py`. It trains a
small two-layer classifier from 13 MFCC coefficients across 97 frames, exports
the model to `kws_model.onnx`, and uses a flattened input of 1261 floats.

```bash
cd ~/xiao_kws_rl
source .venv/bin/activate
python train_ot_and_export.py
```

The older `train_and_export.py` remains in the repository for comparison. It
also trains and exports a compatible 13 x 97 MFCC model, but it is not the
preferred workflow.

After training, move the exported ONNX model into the firmware component:

```bash
mv kws_model.onnx main/kws_model.onnx
```

If `main/kws_model.onnx.data` is created by the exporter, move it with the ONNX
file as well:

```bash
mv kws_model.onnx.data main/kws_model.onnx.data
```

### 5. Generate `kws_model.h` and `kws_model.c`

`onnx_transpiler.py` reads a model with the layer names `fc1.weight`,
`fc1.bias`, `fc2.weight`, and `fc2.bias`, then writes C99 weights and a dense
forward pass. Run it from `main/` so that its fixed relative paths select
`main/kws_model.onnx` and write the generated files into the firmware
component:

```bash
cd ~/xiao_kws_rl/main
source ../.venv/bin/activate
python ../onnx_transpiler.py
```

Important: the current transpiler intentionally generates the original simple
ONNX inference implementation only. The checked-in firmware version of
`kws_model.c` additionally provides `kws_model_init()` and copies the generated
weights into PSRAM before inference. `main/main.c` calls that initializer.
Therefore, do not flash immediately after overwriting `main/kws_model.c` with
the transpiler output: the firmware will fail to build until the generated
weights are integrated into the PSRAM-aware implementation or the transpiler is
updated to emit `kws_model_init()`.

Before replacing the tracked generated files, retain a comparison copy:

```bash
cd ~/xiao_kws_rl
cp main/kws_model.h /tmp/kws_model.previous.h
cp main/kws_model.c /tmp/kws_model.previous.c
cd main
python ../onnx_transpiler.py
diff -u /tmp/kws_model.previous.h kws_model.h || true
diff -u /tmp/kws_model.previous.c kws_model.c || true
```

The generated model must retain these dimensions for the current MFCC and
firmware interfaces:

```text
MODEL_INPUT_SIZE  = 1261  (13 MFCC coefficients x 97 frames)
MODEL_HIDDEN_SIZE = 64
MODEL_OUTPUT_SIZE = 6
```

### 6. Build the firmware

Exit the model virtual environment, activate ESP-IDF, and build from the
repository root:

```bash
deactivate
cd ~/xiao_kws_rl
source ~/.espressif/tools/activate_idf_v6.0.2.sh
idf.py set-target esp32s3
idf.py build
```

For a standard manual ESP-IDF installation, use this activation command
instead:

```bash
source ~/esp/esp-idf/export.sh
idf.py build
```

The successful build creates:

```text
build/esp32_s3_kws_rl.bin
build/esp32_s3_kws_rl.elf
build/bootloader/bootloader.bin
build/partition_table/partition-table.bin
```

Use `idf.py fullclean` before rebuilding only when changing ESP-IDF versions,
target chips, or when CMake configuration becomes inconsistent:

```bash
idf.py fullclean
idf.py set-target esp32s3
idf.py build
```

### 7. Connect, flash, and monitor the XIAO

Connect the XIAO ESP32-S3 Sense using a USB data cable. Find its serial device:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

On the validated Debian setup, the board is `/dev/ttyACM0`. If the current user
cannot access the device, grant access through the standard `dialout` group,
then sign out and back in for the group change to take effect:

```bash
sudo usermod -aG dialout "$USER"
```

Flash the firmware. Replace `/dev/ttyACM0` if your device uses a different
serial path:

```bash
cd ~/xiao_kws_rl
source ~/.espressif/tools/activate_idf_v6.0.2.sh
idf.py -p /dev/ttyACM0 flash
```

Open the serial monitor at the configured 115200 baud rate:

```bash
idf.py -p /dev/ttyACM0 monitor
```

Or build, flash, and monitor in one command:

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

Quit the monitor with `Ctrl+]`. Do not run a second monitor or other serial
program against the same device while it is open.

If flashing reports `Failed to connect to ESP32-S3: No serial data received`,
put the board into bootloader mode: hold `BOOT`, press and release `RESET`, then
release `BOOT`. Run the flash command again. A successful flash reports `Hash
of data verified` for the bootloader, partition table, and application image.

### 8. Confirm runtime audio and inference

After reset, remain quiet during the `1.5` second VAD calibration. A healthy
boot includes lines similar to:

```text
KWS_MODEL: Model weights initialized successfully in PSRAM
MFCC_DSP: MFCC DSP pipeline fully initialized.
KWS_RL_SYSTEM: [VAD] Calibrated: noise=... trigger=...
```

The firmware then logs each VAD capture, MFCC range, raw class logits, and its
top class. The current audio path removes microphone DC offset before energy
measurement and MFCC extraction, allowing ordinary-distance speech to trigger
without treating the microphone bias as ambient sound.

```text
KWS_RL_SYSTEM: [VAD] Triggered at ... RMS (threshold ...)
KWS_RL_SYSTEM: [MODEL] Raw Native Logits: [ ... ]
KWS_RL_SYSTEM: [MATCH SUCCESS] Detected Word: UP (Confidence: ...%)
```

For repeatable checks, speak `UP`, `DOWN`, `LEFT`, `RIGHT`, `BACK`, and `GO` in
that order, with a pause between words. Detection threshold behavior and model
classification accuracy are separate: a VAD trigger confirms capture, while an
incorrect label indicates a model/data mismatch that requires retraining or
model evaluation.

And mainly...HAVE FUN!!!! Sidney ;)

