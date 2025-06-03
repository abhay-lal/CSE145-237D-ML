# 🦉 Burrowing Owl Call Classifier

This repository provides training and evaluation pipelines for deep learning models to classify **burrowing owl vocalizations** using spectrograms derived from audio signals. It includes a custom-designed **TinyCNN** optimized for embedded deployment on STM32 microcontrollers, with optional comparison to baseline models like MobileNetV2 and ProxylessNAS.

---

## 📁 Project Structure

```
CSE145ML-main/
│
├── scripts/
│   ├── train_tinycnn.py             # Train Custom TinyCNN model
│   ├── TinyCNN-to-TfLite.py         # Convert TinyCNN to TFLite and C header
│   ├── Torch_C_header.py            # Convert ProxylessNAS to TFLite + header (benchmark only)
│   ├── tflite_quantize.py           # AI-Edge quantization for benchmarking models
│   ├── test_tflite.py               # Evaluate TFLite models and plot accuracy
│   ├── test.py                      # Evaluate trained models
│   ├── convert_to_tflite.py         # (Legacy) Convert PyTorch to TFLite
│   ├── export_to_onnx.py            # (Deprecated) Convert PyTorch to ONNX
│   └── dataset.py                   # OwlSoundDataset class
│
├── models/                          # Pretrained/exported models
├── graphs/                          # Training/evaluation plots
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🐦 Dataset

This project uses the **BUOWSET** dataset:

* **Classes**: Cluck, Coocoo, Twitter, Alarm, Chick Begging, no_buow  
* **Format**: WAV audio files + metadata CSV  
* **Metadata**: `meta/metadata.csv` with fold info  
* **Splits**: Folds 0-2 = training, Fold 3 = validation  

---

## 🏋️ Training Instructions

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

> 💡 *Windows users should install `soundfile` manually if torchaudio fails.*

### 2. Train Model

```bash
python scripts/train_tinycnn.py
```

This will train the **TinyCNN model**, which is used for final STM32 deployment.

---

## 🔢 Evaluate Model

```bash
python scripts/test.py
```

Outputs:
* Accuracy
* Precision / Recall / F1
* Confusion matrix

Results are saved to the `graphs/` directory.

---

## 📆 Model Export

The Custom-TinyCNN model is exported using the `TinyCNN-to-TfLite.py` script. This script handles the complete flow from a PyTorch `.pth` model to a TFLite `.tflite` file, and generates a C header (`.h`) file for STM32 deployment.

To export the model:

```bash
python TinyCNN-to-TfLite.py
```

This will:
- Load the TinyCNN PyTorch model
- Convert it to TensorFlow Lite format
- Apply post-training quantization
- Generate:
  - `buow_tinycnn.tflite`
  - `buow_tinycnn.h` (for use with TF-Lite Micro on STM32)

---

## 🔧 Model Quantization

Quantization is performed during TFLite conversion using **TensorFlow Lite’s post-training quantization**. Specifically:

- Quantization mode: `tf.lite.Optimize.DEFAULT`
- Resulting model size: ~**11KB**
- Optimized for **low memory and flash usage** suitable for STM32 deployment

Quantization is built into the `TinyCNN-to-TfLite.py` script and requires no additional tooling.

---

## 📲 STM32 Deployment

The final quantized model (`buow_tinycnn.tflite`) is compiled into a C header file (`buow_tinycnn.h`) using:

```bash
xxd -i buow_tinycnn.tflite > buow_tinycnn.h
```

This header can be included directly in embedded C projects using **TF-Lite Micro**.

> This is the **only model used for deployment**.  
> All other models (e.g., MobileNetV2, ProxylessNAS) were used for baseline comparison only and are **not deployed**.

---

## 📊 Sample Results

| Metric   | ProxylessNAS | MobileNetV2 | TinyCNN |
| -------- | ------------ | ----------- | ------- |
| Accuracy | 89.5%        | 88.2%       | 87.1%   |
| F1-Score | 0.89         | 0.87        | 0.86    |

> TinyCNN offers a compact trade-off with deployment feasibility on embedded hardware.

---

## 🧠 Models Used

* **Custom-TinyCNN** (🟢 Deployed): Compact, purpose-built CNN for real-time classification on STM32H747I-DISCO (~11KB with quantization)  
* **MobileNetV2** (🔵 Benchmark only): Lightweight mobile CNN used for baseline performance comparison  
* **ProxylessNAS** (🔵 Benchmark only): NAS-optimized CNN evaluated during early experiments  

---

## 🤖 Authors

* **Abhay Lal** – M.S. CSE, UC San Diego  
* **Zach Lawrence** – B.S. Computer Science, UC San Diego  
* **Max Shen** – B.S. Computer Engineering, UC San Diego  

---

## 📜 License

MIT License. Feel free to use and modify.

---

## 📚 Acknowledgements

Special thanks to the CSE 145/237D Embedded System Design Project course instructors at UC San Diego for project guidance and to the creators of BUOWSET for providing the dataset. Also special mention to Ludwig for initiating this project and being a helpful mentor to work with.
