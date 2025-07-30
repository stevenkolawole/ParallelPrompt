# PARALLELPROMPT: A Benchmark for Intra-Query Semantic Parallelism

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/TBD)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ParallelPrompt-yellow)](https://huggingface.co/datasets/forgelab/ParallelPrompt)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**PARALLELPROMPT** is the first benchmark for measuring intra-query semantic parallelism in real-world LLM prompts. Our benchmark enables both method and system evaluation by providing 37,000+ naturally occurring prompts with structured schemas that reveal parallelizable structure within individual user queries.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/stevenkolawole/parallelprompt.git
cd parallelprompt

# Compile the execution engine
make

# Run a quick test (10 samples)
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output exec_outputs/test.json

# View results
cat exec_outputs/test.json
```

---

## 📊 Key Results

- **10.3%** of real user prompts contain parallelizable structure  
- **1.4–5.7× speedups** across different task categories  
- **>90% quality preservation** on factual tasks  
- **37,000+ prompts** across 11+ languages with structured schemas  

---

## 🏗️ Architecture Overview

PARALLELPROMPT implements a **two-phase architecture** for structure-aware LLM execution:

```
Phase 1: Schema Extraction         Phase 2: Execution Evaluation
┌─────────────────────────┐       ┌────────────────────────┐
│  Raw User Prompt        │ ────► │   Serial Execution     │
│                         │       │   Parallel Execution   │
│ find_parallelprompts.py │       │   Performance Analysis │
│ (Claude 3.5 via Bedrock)│       │                        │
└─────────────────────────┘       │   C++ Execution Suite  │
                                  │   (OpenAI GPT-4)       │
                                  └────────────────────────┘
```

---

### Phase 1: Schema Extraction (`data_curation/`)

- **Input**: Raw user prompts from LMSYS-Chat-1M, WildChat-1M  
- **Process**: Claude 3.5 via AWS Bedrock extracts parallelizable structure  
- **Validation**: Three-tier validation system (high/medium/low confidence)  
- **Output**: Structured CSV with 5-field schemas (template, context, data/n, category)  

---

### Phase 2: Execution Evaluation (`src/`)

- **Input**: Validated schemas from Phase 1  
- **Process**: Category-agnostic parallel execution using OpenAI GPT-4  
- **Metrics**: Latency, quality, normalized speedup analysis  
- **Output**: Performance comparisons, speedup measurements  

---

## 📁 Repository Structure

```text
parallelprompt/
├── data_curation/          # Phase 1: Schema extraction (AWS Bedrock)
│   ├── find_parallelprompts.py    # Claude 3.5 Haiku extraction script
│   ├── system_prompt.txt          # Comprehensive extraction prompt
│   └── README.md                  # Bedrock setup & curation docs
├── datasets/               # Benchmark data (also on HuggingFace)
│   ├── lmsys_parallelizable_queries.csv    # LMSYS subset
│   ├── wildchat_parallelizable_queries.csv # WildChat subset  
│   └── README.md                            # Dataset documentation
├── src/                    # Phase 2: Execution engine (OpenAI)
│   ├── serial_vs_parallel.cpp     # Main benchmarking suite
│   ├── parallel_vary_n.cpp        # Scalability analysis
│   └── Makefile                   # Build system
├── evaluation/             # Quality assessment tools
│   ├── openai_eval/             # LLM judge evaluation
│   └── README.md                # Evaluation documentation
├── utils/                  # Schema conversion utilities
└── include/                # OpenAI API headers
```

---

## 🛠️ Installation & Setup

### Prerequisites

- **C++ Compiler**: GCC 9+ or Clang with C++20 support  
- **Libraries**: `libcurl`, `nlohmann-json`  
- **API Access**:
  - AWS Bedrock (for schema extraction – optional)
  - OpenAI API key (for execution benchmarking)  

### Build Instructions

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential libcurl4-openssl-dev

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Compile engine
make
```

### Optional: Schema Extraction Setup

```bash
pip install boto3 pandas tqdm datasets backoff

# Set AWS credentials
export AWS_KEY="your-aws-access-key"
export AWS_SECRET_KEY="your-aws-secret-key"
```

---

## 📖 Usage Guide

### Basic Execution

```bash
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output exec_outputs/test.json
```

### Advanced Options

```bash
# Run with 50 samples
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output exec_outputs/sample.json --sample-size 50 --post-process
```

### Output Format

```json
{
  "prompt": "Generate 10 room descriptions...",
  "category": "Repeated Generation",
  "serial_output": "...",
  "parallel_output": ["...", "...", "..."],
  "speedup": 3.41,
  "normalized_speedup": 4.22,
  "serial_duration_ms": 5420,
  "total_parallel_duration_ms": 1590
}
```

---

## 🎯 Use Cases

### 1. Schema Extraction Evaluation

```bash```
<!-- python your_method.py --input raw_prompts.txt --output your_schemas.csv
./bin/alphabits --queries your_schemas.csv --output comparison.json
``` -->

### 2. Execution Strategy Benchmarking

```bash
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output baseline.json
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output enhanced.json --post-process
```

### 3. Model Benchmarking

- Modify `src/serial_vs_parallel.cpp`
- Compare models on serial vs. parallel execution

---

## 📊 Performance Metrics

- **Raw Speedup**: Latency comparison  
- **Normalized Speedup**: Length-normalized throughput  
- **Quality Preservation**: Semantic equivalence via LLM judge  
- **Category-wise Trends**: Per-task analysis  

---

## 🔧 Extending the Benchmark

### Add New Categories

1. Update `system_prompt.txt` in `data_curation/`  
2. Add parsing logic  
3. Regenerate schemas  

### Add Evaluation Metrics

1. Modify `main()` in C++  
2. Update `evaluation/` scripts  
3. Re-run experiments  

---

## 📚 Citation

If you use this benchmark, please cite:

```bibtex
@article{parallelprompt2025,
  title={PARALLELPROMPT: Extracting Parallelism from Large Language Model Queries},
  author={To be Updated},
  journal={To be Updated},
  year={2025}
}
```

---

## 🤝 Contributing

We welcome contributions!

```bash
git clone https://github.com/your-username/parallelprompt.git
cd parallelprompt
git checkout -b feature/my-feature
make clean && make
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output test.json
```

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for full details.

---

## 📄 License

This project is licensed under the MIT License – see [`LICENSE`](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Dataset Sources**: LMSYS-Chat-1M, WildChat-1M  
- **Model APIs**: OpenAI GPT-4, Claude 3.5  
<!-- - **Community**: Thanks to all contributors and researchers using PARALLELPROMPT   -->

---

## 💬 Questions?

- 🐛 [GitHub Issues](https://github.com/stevenkolawole/parallelprompt/issues)  
