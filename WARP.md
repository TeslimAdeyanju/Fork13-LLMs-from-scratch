# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository implements a GPT-like Large Language Model from scratch in PyTorch, following the book "Build a Large Language Model (From Scratch)" by Sebastian Raschka. The codebase is organized by chapters, with each chapter building upon the previous ones to implement a complete LLM training pipeline.

## Development Commands

### Environment Setup
```bash
# Install dependencies using pip
pip install -r requirements.txt

# Or using uv (recommended for faster installs)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --dev --python=3.10

# Or using pixi (conda alternative)
pixi install
```

### Testing
```bash
# Run specific chapter tests
pytest --ruff ch04/01_main-chapter-code/tests.py
pytest --ruff ch05/01_main-chapter-code/tests.py
pytest --ruff ch06/01_main-chapter-code/tests.py

# Test Jupyter notebooks
pytest --ruff --nbval ch02/01_main-chapter-code/dataloader.ipynb
pytest --ruff --nbval ch03/01_main-chapter-code/multihead-attention.ipynb

# Run all package tests
pytest pkg/llms_from_scratch/tests/

# Run bonus material tests
pytest ch02/05_bpe-from-scratch/tests/tests.py
```

### Running the Code
```bash
# Chapter-specific execution
cd ch04/01_main-chapter-code && python gpt.py
cd ch05/01_main-chapter-code && python gpt_train.py
cd ch05/01_main-chapter-code && python gpt_generate.py

# Run Jupyter notebooks
jupyter lab
# Then navigate to chapter notebooks like ch02/01_main-chapter-code/ch02.ipynb
```

### Code Quality
```bash
# Linting (automatically included with pytest --ruff)
pytest --ruff <file_name>

# Validate notebooks
pytest --nbval <notebook_name>.ipynb
```

## Code Architecture

### Core Components Hierarchy

1. **Token Processing (Ch2)**: Text tokenization using tiktoken (GPT-2 tokenizer)
   - `GPTDatasetV1`: Sliding window dataset for training data
   - `create_dataloader_v1`: DataLoader creation with configurable batch size and context length

2. **Attention Mechanism (Ch3)**: Multi-head self-attention implementation
   - `MultiHeadAttention`: Core attention mechanism with causal masking
   - Implements scaled dot-product attention with multiple heads
   - Uses causal masking for autoregressive generation

3. **GPT Model Architecture (Ch4)**: Complete transformer architecture
   - `GPTModel`: Main model class combining embeddings, transformer blocks, and output head
   - `TransformerBlock`: Single transformer layer with attention + feed-forward + residual connections
   - `FeedForward`: MLP with GELU activation
   - `LayerNorm`: Pre-norm architecture following GPT-2 design

4. **Training Pipeline (Ch5)**: Model pretraining infrastructure
   - Loss calculation functions (`calc_loss_batch`, `calc_loss_loader`)
   - Training loop with evaluation (`train_model_simple`)
   - Text generation utilities (`generate_text_simple`)

5. **Finetuning (Ch6-7)**: Task-specific adaptation
   - Classification finetuning (Ch6)
   - Instruction following finetuning (Ch7)
   - Model evaluation utilities

### Key Design Patterns

- **Chapter-based progression**: Each chapter builds upon `previous_chapters.py` imports
- **Configuration-driven**: Models use dictionary-based configuration (e.g., `gpt_config`)
- **Device-agnostic**: Automatic GPU detection and usage when available
- **Modular components**: Each component can be imported and used independently

### Model Configuration Structure
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of transformer layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-key-value bias
}
```

### Directory Structure Logic

- `ch0X/01_main-chapter-code/`: Core chapter implementation and notebooks
- `ch0X/0Y_bonus_*/`: Additional experimental features and optimizations
- `appendix-*/`: Supplementary materials (PyTorch intro, training enhancements, LoRA)
- `pkg/`: Installable Python package for importing utilities
- `setup/`: Installation and environment setup guides

## Development Guidelines

### Testing Philosophy
- Each chapter has a `tests.py` file with basic functionality tests
- Tests use `capsys` to verify expected outputs match exactly
- Notebooks are validated using `nbval` to ensure they execute without errors
- Use `--ruff` flag for automatic linting during testing

### Python Version Requirements
- Primary development: Python 3.10+
- Backward compatibility tested with Python 3.10 in CI
- PyTorch >= 2.3.0 required for compatibility

### GPU Usage
- Code automatically detects and uses CUDA if available
- Falls back to CPU execution for compatibility
- No special configuration needed for GPU acceleration

### Chapter Dependencies
- Later chapters import from `previous_chapters.py` modules
- Maintains chapter-by-chapter learning progression
- Each chapter can be run independently after setup

### Package Management
- Three supported approaches: pip, uv (fastest), or pixi (conda-based)
- `requirements.txt` lists minimum versions with compatibility notes
- `pyproject.toml` defines the installable package structure
- `pixi.toml` provides conda-forge based environment

## Common Operations

### Generate Text with Pretrained Model
```bash
cd ch05/01_main-chapter-code
python gpt_generate.py
```

### Train a Model from Scratch
```bash
cd ch05/01_main-chapter-code
python gpt_train.py
```

### Finetune for Classification
```bash
cd ch06/01_main-chapter-code
python gpt_class_finetune.py
```

### Convert GPT to Llama Architecture
```bash
cd ch05/07_gpt_to_llama
jupyter lab converting-gpt-to-llama2.ipynb
```

### Run Web Interface
```bash
cd ch05/06_user_interface
python app_own.py  # For your own trained model
python app_orig.py # For pretrained model
```

## Hardware Requirements

- **Minimum**: Standard laptop/desktop (tested on M3 MacBook Air)
- **Recommended**: NVIDIA GPU for faster training (chapters 5-7)
- **Memory**: Varies by model size and batch size
- **Storage**: ~1GB for repository + datasets

## Key Files for Understanding

- `ch04/01_main-chapter-code/gpt.py`: Complete GPT implementation
- `ch05/01_main-chapter-code/gpt_train.py`: Training pipeline
- `ch05/01_main-chapter-code/previous_chapters.py`: Consolidated utilities
- `pyproject.toml`: Package configuration and dependencies
- `requirements.txt`: Exact dependency versions