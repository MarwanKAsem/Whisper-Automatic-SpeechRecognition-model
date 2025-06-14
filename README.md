# 🧠 Whisper Fine-Tuning on AMI IHM (English)

This repository contains the full pipeline to fine-tune OpenAI's Whisper model on the **AMI IHM English** dataset using Hugging Face 🤗 `transformers` and `datasets`.

Whisper is a robust speech recognition model capable of multilingual and multitask ASR. Here, we adapt it to the **AMI Individual Headset Microphone (IHM)** English recordings to optimize transcription accuracy in conversational meeting settings.

---

## 🗂️ Contents

- [🔧 Project Overview](#-project-overview)
- [🧰 Installation](#-installation)
- [📚 Dataset](#-dataset)
- [🧹 Preprocessing](#-preprocessing)
- [🏋️‍♂️ Training](#-training)
- [📊 Evaluation](#-evaluation)
- [📁 Directory Structure](#-directory-structure)
- [🚀 Model Upload (Hugging Face Hub)](#-model-upload-hugging-face-hub)
- [📜 License](#-license)
- [🙌 Acknowledgements](#-acknowledgements)

---

## 🔧 Project Overview

- **Model:** `openai/whisper-small`
- **Language:** English
- **Framework:** Hugging Face Transformers
- **Training Goal:** Reduce Word Error Rate (WER)
- **Dataset:** [AMI IHM](https://huggingface.co/datasets/edinburghcstr/ami)
- **Training Strategy:** Epoch-based evaluation, mixed precision (fp16), fast optimization settings

---

## 🧰 Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/whisper-ami-ihm
cd whisper-ami-ihm
pip install -r requirements.txt
```
## 🚀 Getting Started

### Clone the Repository
```bash
/git clone https://github.com/MarwanKAsem/socialmedia-Text-Generation-model.git
cd whisper-small-hi2
```
## Install Dependencies
```
pip install -r requirements.txt
```
## Prepare the Dataset
We use the AMI Corpus - IHM (Individual Headset Microphone) via Hugging Face:
```
from datasets import load_dataset

dataset = load_dataset(
    "edinburghcstr/ami",
    "ihm",
    split="train+validation+test"
)

print(dataset[0])
```
⚠️ This version includes speaker-separated headset audio and manual transcripts.


## Training
We fine-tune openai/whisper-small using the 🤗 Seq2SeqTrainer. You can modify parameters in train.py.
Run your script  using Trainer with a Whisper model and dataset to begin training.

## 📊 Evaluation
We evaluate using Word Error Rate (WER) with jiwer:
Evaluation Metric: Word Error Rate (WER)
Best WER Achieved: (update after training completes)

## 📁 Directory Structure
whisper-small-hi2/
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── config/                  # Training configs (optional)
├── data/                    # Local dataset storage (optional)
├── outputs/                 # Saved models and logs
├── requirements.txt         # Dependencies
└── README.md

## ✨ Features

Optimized for fast training

Supports fp16 and gradient checkpointing

Hugging Face Hub integration ready

Compatible with Datasets and Trainer

## 📜 License
This project is licensed under the MIT License. Feel free to use, modify, and share!

## 🙌✨ Acknowledgements
Hugging Face 🤗
OpenAI Whisper
AMI Meeting Corpus (AMI IHM Dataset ==> https://huggingface.co/datasets/edinburghcstr/ami )
📝 License
This project is licensed under the MIT License. Dataset usage must comply with AMI Corpus terms.
