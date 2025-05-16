# Data Curation Pipeline

This directory contains the tools and resources used to construct the PARALLELPROMPT benchmark by extracting parallelizable prompts from large-scale LLM conversation datasets.

## Overview

The data curation pipeline identifies and validates prompts with latent parallelizable structure, implementing the multi-stage filtering process described in our paper:

1. **First-turn extraction** from conversation logs
2. **LLM-assisted classification** and schema extraction
3. **Structured validation** with confidence tier assignment

Through this process, we identified over 37,000 naturally occurring parallelizable prompts spanning multiple languages and task categories.

## Contents

- `find_parallelprompts.py`: Core script for identifying parallelizable prompts and extracting structured schemas
- `run_finder.sh`: Wrapper script for processing datasets with AWS Bedrock API
- `system_prompt.txt`: Carefully designed prompt for LLM-based classification and schema extraction
- `stats/`: Contains validation statistics from our curation process
 - `lmsys_validation_stats.json`: Statistics from LMSYS-Chat-1M
 - `wildchat_validation_stats.json`: Statistics from WildChat-1M

## Requirements

- Python 3.8+
- Required Python packages: `pandas`, `tqdm`, `boto3`, `datasets`, `backoff`
- AWS credentials with access to Bedrock API
- Access to source datasets:
 - [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/chat-1m)
 - [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)

## Usage

### Setup

1. Set up AWS credentials with access to Claude 3.5 Haiku via Bedrock:

```bash
export AWS_KEY="your_aws_key"
export AWS_SECRET_KEY="your_aws_secret_key"
export AWS_REGION="your_aws_region"  # Default: us-east-1
```

2. Install required packages:

```
pip install pandas tqdm boto3 datasets backoff
```

### Running the Pipeline

You can process conversation datasets directly from Hugging Face:

```
# Using the wrapper script
./run_finder.sh

# Or specifying a dataset directly
python find_parallelprompts.py --dataset lmsys/lmsys-chat-1m
```

The script will:

- Load the dataset from Hugging Face
- Extract first-turn user messages
- Process them in batches using Claude 3.5 Haiku via AWS Bedrock
- Apply structured validation rules
- Save parallelizable prompts with their extracted schemas
- Track validation statistics and novel categories

## Pipeline Architecture

The pipeline uses a multi-stage approach:

- **Extraction**: Extracts user prompts from conversation datasets
- **Schema Classification**: Uses Claude 3.5 to identify parallelizable prompts and extract structured schemas
- **Validation**: Applies rule-based checks to verify schema integrity
- **Confidence Assignment**: Categorizes schemas into high/medium confidence tiers based on structural indicators

### Validation Tiers

Prompts are assigned confidence tiers based on structural indicators:

- **High Confidence**: Prompts with explicit structural markers like numbered lists or clear multiplicity indicators
- **Medium Confidence**: Prompts with softer structural cues like comma-separated lists or plural forms
- **Failed Validation**: Prompts that don't meet criteria for parallelizable structure

## Schema Structure

Successful schema extraction produces a JSON object with the following structure:
```{
  "`serial`": "Original user prompt",
  "`template`": "Task template with {data} placeholders",
  "context`": "Shared content across subtasks",
  "`data`": ["item1", "item2", "item3"],  // OR
  "`n`": 5,  // Number of generations (mutually exclusive with data)
  "`category`": "Reading Comprehension",
  "`validation_tier`": "high_confidence"
}
```
## System Prompt Design
The `system_prompt.txt` file contains a carefully crafted prompt that instructs Claude to:

- Identify if a prompt contains parallelizable structure
- Extract the task template with appropriate placeholders
- Identify any shared context across subtasks
- Extract either a list of data items or a generation count
- Classify the prompt into our taxonomy of parallelizable patterns

The prompt enforces mutual exclusivity constraints (data or n, but never both) and guides extraction toward canonical categories while allowing novel patterns to emerge.

## Statistics and Monitoring
The pipeline tracks extensive statistics about the extraction process:

- Category distribution
- Validation success rates by category
- High vs. medium confidence distribution
- Novel category emergence and frequency
- Failure patterns and rates

These statistics allow monitoring of the curation process and provide insights into the distribution of parallelizable patterns in real-world LLM interactions.

## Extending the Pipeline
To adapt this pipeline for new datasets or use cases:

- Modify `system_prompt.txt` to include additional examples or categories
- Adjust validation rules in `find_parallelprompts.py` as needed
- Update the confidence tier criteria for your specific needs

## Citation
If you use this data curation pipeline in your research, please cite our paper:
```@misc{parallelPrompt2025,
  author = {forgelab},
  title = {PARALLELPROMPT: Extracting Parallelism from Large Language Model Queries},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/forgelab/ParallelPrompt}}
}```