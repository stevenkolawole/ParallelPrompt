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

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Compile the execution engine
make

# Run a quick test (10 samples by default, schema-driven)
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output test_results.json

# Run end-to-end evaluation (with schema extraction)
./bin/alphabits --queries datasets/wildchat_parallelizable_queries.csv --output e2e_results.json --end-to-end

# View results
cat test_results.json
```

---

## 📊 Key Results

- **10.3%** of real user prompts contain parallelizable structure  
- **Up to 7x speedups** across different task categories  
- **>90% quality preservation** on factual tasks  
- **37,000+ prompts** across 11+ languages with structured schemas  

---

## 🏗️ Architecture Overview

PARALLELPROMPT supports both **schema-driven** and **end-to-end** execution modes:

### Schema-Driven Mode (Default)

```
Pre-extracted Schemas (CSV)  ──►  Parallel Execution Engine (C++)
                                  ├─ Serial Execution
                                  ├─ Parallel Execution  
                                  └─ Performance Analysis
```

### End-to-End Mode (`--end-to-end`)

```
Raw User Prompt  ──►  Schema Extraction  ──►  Parallel Execution Engine
                      (GPT-4o)                ├─ Serial Execution
                                              ├─ Parallel Execution
                                              └─ Performance Analysis
```

---

## 📁 Repository Structure

```text
parallelprompt/
├── src/                    # Execution engine (C++)
│   ├── serial_vs_parallel.cpp     # Main benchmarking suite
│   │                              # - Schema-driven execution
│   │                              # - End-to-end evaluation
│   │                              # - Post-processing support
│   ├── parallel_vary_n.cpp        # Scalability analysis
│   └── Makefile                   # Build system
├── datasets/               # Benchmark data (Detailed documentation on HuggingFace's Dataset link)
│   ├── lmsys_parallelizable_queries.csv    # LMSYS subset (963 prompts)
│   ├── wildchat_parallelizable_queries.csv # WildChat subset
├── data_curation/          # Schema extraction tools (legacy)
│   ├── find_parallelprompts.py    # Original Claude 3.5 extraction
│   └── system_prompt.txt          # Extraction prompt template
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
- **OpenAI API Key**: For both schema extraction (if doing end-to-end) and execution

### Build Instructions

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential libcurl4-openssl-dev

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Compile the execution engine
make

# Verify installation
./bin/alphabits --help
```

---

## 📖 Usage Guide

### Schema-Driven Execution (Recommended for Testing)

Uses pre-extracted schemas from the CSV files for fast evaluation:

```bash
# Basic execution (10 samples)
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output results.json

# Custom sample size
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output results.json --sample-size 50

# Full dataset
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output full_results.json --sample-size all

# With post-processing cleanup
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output clean_results.json --post-process
```

### End-to-End Evaluation

Extracts schemas from raw prompts using GPT-4o, then executes in parallel:

```bash
# End-to-end with schema extraction
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output e2e_results.json --end-to-end

# End-to-end with post-processing
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output e2e_clean.json --end-to-end --post-process

# Small sample for testing
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output e2e_test.json --sample-size 5 --end-to-end
```

### Command Line Options

| Option            | Description                                         | Default |
|-------------------|-----------------------------------------------------|---------|
| `--queries`       | Path to CSV file with prompts                       | Required |
| `--output`        | Output JSON file path                               | Required |
| `--sample-size`   | Number of prompts to process (`<num>` or `all`)     | 10 |
| `--post-process`  | Enable output cleanup using GPT-4o-mini             | Disabled |
| `--end-to-end`    | Extract schemas from raw prompts (vs. using CSV)    | Disabled |

---

## 📊 Output Format

### Schema-Driven Mode

```json
{
  "prompt": "Generate 10 room descriptions...",
  "category": "Repeated Generation",
  "serial_output": "...",
  "parallel_output": ["...", "...", "..."],
  "speedup": 3.41,
  "normalized_speedup": 4.22,
  "serial_duration_ms": 5420,
  "total_parallel_duration_ms": 1590,
  "post_processed_output": "..." // if --post-process enabled
}
```

### End-to-End Mode

```json
{
  "prompt": "Generate 10 room descriptions...",
  "category": "Repeated Generation",
  "extracted_category": "Repeated Generation",
  "extracted_template": "Generate a detailed description of {data}...",
  "schema_extraction_duration_ms": 1200,
  "e2e_parallel_duration_ms": 2790,
  "e2e_speedup": 1.94,
  "extraction_successful": true,
  // ... plus all schema-driven fields
}
```

---

## 🔧 Extending the Benchmark

### Benchmarking Parallelization Methods

```bash
./bin/alphabits --queries datasets/lmsys_parallelizable_queries.csv --output baseline.json

# Your decomposition method
./bin/alphabits --queries your_schemas.csv --output your_results.json
```

### Try Other Models (for Model Performance Analysis)

Update model strings in:

```cpp
call_openai(..., model="gpt-4o-mini", ...)
call_gpt_schema_extraction(..., model="gpt-4o", ...)
```

### Add Custom Categories

1. Update `extract_schema_from_prompt()`  
2. Extend `get_system_prompt()` for new cases  
3. Evaluate with `--end-to-end`

### Custom Post-Processing

Edit `post_process_outputs()` in `src/serial_vs_parallel.cpp`:

```cpp
string post_process_prompt = "Your custom post-processing instructions...";
```

---

## 📚 Citation

If you use this benchmark, please cite:

```bibtex
@article{parallelprompt2025,
  title={PARALLELPROMPT: Extracting Parallelism from Large Language Model Queries},
  author={TBD},
  journal={arXiv preprint arXiv:TBD},
  year={2025}
}
```

---

## 💬 Questions?

- 🐛 [GitHub Issues](https://github.com/stevenkolawole/parallelprompt/issues)  
- 📧 [Contact](TBD)
<!-- (mailto:skolawol@andrew.cmu.edu) -->
