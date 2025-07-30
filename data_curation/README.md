# Data Curation Pipeline

This directory contains the **Phase 1** schema extraction pipeline that identifies parallelizable prompts from raw LLM conversation logs using Claude 3.5 Haiku via AWS Bedrock.

---

## 🎯 Overview

The curation pipeline transforms raw user prompts into structured schemas that enable parallel execution:

```
Raw Prompts → Claude 3.5 Haiku (Bedrock) → Schema Extraction → Validation → Benchmark Dataset
```

---

## 📁 Files

- `find_parallelprompts.py`: Main extraction script using Claude 3.5 Haiku via AWS Bedrock  
- `system_prompt.txt`: Comprehensive prompt for schema extraction with explicit rules  
- `run_finder.sh`: Batch processing script for the main extraction script  
- `stats/`: Validation statistics and analysis results  

---

## ⚙️ AWS Bedrock Setup

### Prerequisites

```bash
pip install boto3 pandas tqdm datasets backoff

export AWS_KEY="your-aws-access-key"
export AWS_SECRET_KEY="your-aws-secret-key"
export AWS_REGION="us-east-1"  # Optional, defaults to us-east-1
```

### Model Configuration

We use **Claude 3.5 Haiku** (`us.anthropic.claude-3-5-haiku-20241022-v1:0`) for low-latency, cost-efficient schema extraction with strong JSON compliance.

---

## 🏗️ Schema Format

Each validated prompt yields a 5-field schema:

```python
{
  "template": "Translate: {data}",
  "context": "Translate to French",
  "data": ["Hello", "Goodbye", "Thanks"],  # OR use "n": 10
  "n": 10,
  "category": "Translation"
}
```

---

## 🚀 Usage

### Basic Extraction

```bash
python find_parallelprompts.py --dataset allenai/WildChat-1M # default is lmsys/lmsys-chat-1m
 
```

---

## 🎨 System Prompt Design

### 1. Pattern Recognition

- Parallelism indicators: "list", "generate", "each", "every"
- Linguistic structures: bullet points, plural markers
- Numerical cues: "10 descriptions", "3 examples"

### 2. Schema Generation

- Structured JSON output
- Mutual exclusivity rules (data XOR n)
- Explicit template-placeholder alignment

### 3. Quality Control

- Min parallelism length: ≥2
- Template must include `{data}` if `data` is used
- Validation enforces consistency and clarity

---

## 📊 Validation Framework

### Three-Tier Confidence System

| Tier            | Description                          |
|-----------------|--------------------------------------|
| High (62%)      | Strong structural cues               |
| Medium (38%)    | Semantic patterns, weaker cues       |
| Failed          | Template violations or ambiguity     |

### Validation Logic (Simplified)

```python
def validate_schema(schema):
    if schema.has_data and schema.has_n:
        return False
    if schema.has_data and "{data}" not in schema.template:
        return False
    if schema.has_data and len(schema.data) < 2:
        return False
    return True
```

---

## 🌍 Multilingual Support

The extraction pipeline supports 11+ languages. Success rates vary by structure and formatting norms.

| Language  | Success Rate | Notes                        |
|-----------|--------------|------------------------------|
| English   | 62%          | All categories covered       |
| Chinese   | 28%          | Strong in NER (35%)          |
| Russian   | 55%          | Frequent in language editing |
| Japanese  | 34%          | Translation-focused prompts  |

---

## 📈 Performance Metrics

### Example Validation Summary

```json
{
  "total_processed": 173000,
  "high_confidence": 22953,
  "medium_confidence": 14068,
  "failed_validation": 136000,
  "categories": {
    "Reading Comprehension": 12744,
    "Repeated Generation": 10915,
    "Named Entity Recognition": 4449
  }
}
```

---

## 🔧 Customization

### Add New Categories

1. Add examples to `system_prompt.txt`  
2. Include category descriptor and markers  
3. Validate with test batch  

```python
new_category = {
  "name": "Code Generation",
  "description": "Write N functions, scripts, or modules",
  "examples": ["Write 5 Python functions...", "Generate 3 CLI apps..."],
  "markers": ["function", "script", "example"]
}
```

### Tweak Validation Thresholds

```python
CONFIDENCE_THRESHOLD = 0.7     # Raise for stricter filtering
MIN_PARALLELISM = 2            # Require longer data lists
MAX_TEMPLATE_LENGTH = 500      # Avoid overlong templates
```

---

## 🔍 Research Applications

- **Prompt Engineering**: Design extraction prompts with higher recall  
- **LLM Comparison**: Use this pipeline with open models (e.g. Mistral, GPT-J)  
- **Cross-Lingual Benchmarking**: Improve support for non-English prompts  
- **Parallel Pattern Discovery**: Identify new candidate structures for parallelism  

---

For execution and evaluation, see the main [README](../README.md)
