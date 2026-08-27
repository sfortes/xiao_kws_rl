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
 - FreeRTOS Queue (g_inference_queue)     <-- Thread-safe message channel Core 0 -> Core 1
 
 CORE 0 TASK (Audio Sampling & Model Execution)
 1. Reads raw 16kHz PCM audio stream via PDM hardware peripheral (GPIO 41 CLK, GPIO 42 DAT)
 2. Extracts MFCC features (13 coefficients x 63 time frames = 8192 floats)
 3. Locks Mutex -> Reads SRAM weights -> Executes C99 ONNX forward pass -> Unlocks Mutex
 4. Applies Softmax to produce probabilities for "UP", "DOWN", "LEFT", "RIGHT", "BACK", "GO"
 5. Pushes {predicted_class, confidence, hidden_activations} to Queue
 
 CORE 1 TASK (Online Reinforcement Learning Update)
 1. Blocks on Queue waiting for inference payload from Core 0
 2. Evaluates environment feedback / confidence metric to assign Reward (+1.0 or -1.0)
 3. Computes policy gradient update on final classification layer: ΔW = lr * reward * activation
 4. Locks Mutex -> Modifies `g_model_weights` directly in SRAM -> Unlocks Mutex


