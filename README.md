# BLOOM-7B LoRA Fine-tuning Experiment 🚀

A learning project to understand Parameter-Efficient Fine-Tuning (PEFT) using LoRA on the BLOOM-7B language model.

## 📋 Overview

This notebook demonstrates how to fine-tune a 7-billion parameter language model using **Low-Rank Adaptation (LoRA)**, making it possible to train large models on consumer-grade GPUs like Google Colab's free tier.

**Task**: Fine-tune BLOOM-7B to generate tags for English quotes

## 🎯 Learning Goals

- Understand how LoRA works for efficient fine-tuning
- Learn to work with quantized models (8-bit precision)
- Practice using HuggingFace Transformers and PEFT libraries
- Explore training large language models with limited resources

## 🧪 Experiment Details

### Model
- **Base Model**: BLOOM-7B1 (7.1 billion parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 8-bit (reduces memory from ~14GB to ~7GB)
- **Trainable Parameters**: Only 0.11% (7.8M out of 7B parameters)

### Dataset
- **Source**: [Abirate/english_quotes](https://huggingface.co/datasets/Abirate/english_quotes)
- **Size**: 2,508 quotes with tags
- **Format**: `"quote" ->: ['tag1', 'tag2', ...]`

### Training Configuration
```python
LoRA Config:
  - rank (r): 16
  - alpha: 32
  - dropout: 0.05
  
Training:
  - Batch size: 1 (effective: 16 with gradient accumulation)
  - Steps: 200
  - Learning rate: 2e-4
  - Precision: FP16 (mixed precision)
```

## 🛠️ Requirements

### Hardware
- **Minimum**: GPU with 15GB VRAM (e.g., Tesla T4)
- **Recommended**: 40GB VRAM (e.g., A100)
- **Platform**: Google Colab (free tier works!)

### Software
```bash
pip install -q bitsandbytes datasets accelerate loralib
pip install -q git+https://github.com/huggingface/transformers.git@main
pip install -q git+https://github.com/huggingface/peft.git
```
## 🧠 Key Concepts Learned

### What is LoRA?
Instead of updating all 7 billion parameters, LoRA:
1. Freezes the original model weights
2. Adds small trainable matrices (rank decomposition)
3. Only trains ~7.8M parameters (0.11%)

**Math**: `W_new = W_frozen + (alpha/r) × B × A`

Where B and A are small matrices we train.

### Why 8-bit Quantization?
- Reduces memory: 14GB → 7GB
- Maintains ~99% performance
- Enables training on consumer GPUs

### Why This Works
- **Efficiency**: Uses minimal memory and compute
- **Quality**: Achieves good results with small adapter
- **Flexibility**: Can train multiple task-specific adapters

## 🤗 Pre-trained Model Available

The fine-tuned LoRA adapters from this notebook are **already available** on HuggingFace Hub:

**🔗 Model**: [issabachar/bloom-7b1-quotes-lora](https://huggingface.co/issabachar/bloom-7b1-quotes-lora)

You can use the trained model directly without re-training:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")

# Load fine-tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, "issabachar/bloom-7b1-quotes-lora")

# Generate
prompt = "Life is beautiful ->:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**Model Details:**
- 📦 Size: ~31.5MB (LoRA adapters only)
- 🎯 Task: Quote tag generation
- 📊 Training: 200 steps on 2,508 English quotes
- 🔧 Base Model: BLOOM-7B1
  
## 🤝 Contributing

This is a personal learning project, but feel free to:
- Open issues for questions or bugs
- Suggest improvements
- Share your own experiments
