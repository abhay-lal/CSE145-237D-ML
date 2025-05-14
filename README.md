# 🦩 Burrowing Owl Call Classifier

This repository provides training and evaluation pipelines for deep learning models to classify **burrowing owl vocalizations** using spectrograms derived from audio signals. It supports lightweight model architectures like MobileNetV2 and ProxylessNAS and enables export to ONNX and TFLite formats for deployment on embedded devices.

---

## 📁 Project Structure

```
CSE145ML-main/
│
├── scripts/
│   ├── train_mobilenetv2.py         # Train MobileNetV2 model
│   ├── train_proxylessnas.py        # Train ProxylessNAS model
│   ├── train_proxylessnas_abhay.py  # Custom ProxylessNAS variant
│   ├── test.py                      # Evaluate trained models
│   ├── convert_to_tflite.py         # Convert PyTorch to TFLite
│   ├── export_to_onnx.py            # Convert PyTorch to ONNX
│   ├── codegen.ipynb                # TinyML deployment notebook
│   └── dataset.py                   # OwlSoundDataset class
│
├── models/                          # Pretrained/exported models
│
├── graphs/                          # Training/evaluation plots
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🐦 Dataset

This project uses the **BUOWSET** dataset:

* **Classes**: Cluck, Coocoo, Twitter, Alarm, Chick Begging, no\_buow
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

### 2. Train Models

#### ProxylessNAS:

```bash
python scripts/train_proxylessnas.py
```

#### MobileNetV2:

```bash
python scripts/train_mobilenetv2.py
```

---

## 🔢 Evaluate Models

```bash
python scripts/test.py
```

Outputs:

* Accuracy
* Precision / Recall / F1
* Confusion matrix

Results are saved to the `Images/` directory.

---

## 📆 Model Export

### Export to ONNX

```bash
python scripts/export_to_onnx.py
```

### Convert to TFLite

```bash
python scripts/convert_to_tflite.py
```

---

## Model Quantization

This project uses the relatively new [AI-Edge-Torch](https://github.com/google-ai-edge/ai-edge-torch) library by Google to convert PyTorch models to TFLite models, with options for `int8` quantization. Package management becomes unmanageable easily with this library, so I recommend the following:

Create a fresh virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```
Install the required dependencies. Note that this may take a long time (~20 min):

```bash
pip install -r https://raw.githubusercontent.com/google-ai-edge/ai-edge-torch/main/requirements.txt
pip install ai-edge-torch-nightly
```

Convert and quantize the PyTorch model:

```bash
python tflite_quantize.py -model_type={ProxylessNAS OR MobileNetV2\} -model_path={PATH_TO_MODEL_PTH\} -target_path={PATH_TO_EXPORT_TO}
```
This will produce two files, consisting of the PyTorch model converted to TFLite format with and without int8 quantization. 

---
## 📊 Sample Results

| Metric   | ProxylessNAS | MobileNetV2 |
| -------- | ------------ | ----------- |
| Accuracy | 89.5%        | 88.2%       |
| F1-Score | 0.89         | 0.87        |

Visual results in `Images/precision_recall_f1.png`

---

## 🧠 Models Used

* **MobileNetV2**: Efficient CNN for mobile inference
* **ProxylessNAS**: NAS-optimized CNN for embedded applications

Model formats:

* `.pth` (PyTorch)
* `.onnx` (ONNX)
* `.tflite` (TensorFlow Lite)

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

Special thanks to the CSE 145/237D Embedded System Design Project course instructors at UC San Diego for project guidance and to the creators of BUOWSET for providing the dataset.Also special mention to Ludwig for initiating this project and being a helpful mentor to work with. 
