import pandas as pd
import os
import time
import re
from tqdm import tqdm
import json
import boto3
from datasets import load_dataset
import csv
import uuid
from concurrent.futures import ThreadPoolExecutor
import backoff
import argparse

AWS_KEY = os.getenv("AWS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

if not AWS_KEY or not AWS_SECRET_KEY:
    raise ValueError("Missing AWS credentials. Please set AWS_KEY and AWS_SECRET_KEY as env vars.")

# Configure AWS Bedrock
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_KEY, 
    aws_secret_access_key=AWS_SECRET_KEY
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, default="lmsys/lmsys-chat-1m",
                        help="HuggingFace dataset name, e.g., lmsys/lmsys-chat-1m or allenai/WildChat-1M")
    return parser.parse_args()

dataset_name = parse_args().dataset
prefix = dataset_name.split("-")[0].split("/")[1].lower()

# Load the dataset
print(f"Loading {dataset_name} ...")
dataset = load_dataset(dataset_name)

# Output file setup
field_names = ["index", "query_id", "prompt", "parallelizable", "category", "is_novel_category", "category_description", 
               "serial", "template", "context", "data", "n", "validation_passed", "validation_tier", "timestamp"]
output_file = f"{prefix}_parallelizable_queries.csv"
validation_stats_file = f"{prefix}_validation_stats.json"

# Setup for tracking novel categories in a separate file
novel_categories_file = f"{prefix}_novel_categories.json"
novel_categories_lock = {}

# Initialize CSV if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, 
            fieldnames=field_names,
            quoting=csv.QUOTE_ALL,
            quotechar='"',
            escapechar='\\'
        )
        writer.writeheader()

# Initialize novel categories tracking file
if os.path.exists(novel_categories_file):
    with open(novel_categories_file, 'r') as f:
        try:
            novel_categories_lock = json.load(f)
        except json.JSONDecodeError:
            novel_categories_lock = {}
else:
    with open(novel_categories_file, 'w') as f:
        json.dump({}, f)

# Initialize validation stats tracking
validation_stats = {
    "total_classified_as_parallelizable": 0,
    "failed_validation": 0,
    "passed_validation": 0,
    "categories_failed": {},
    "categories_passed": {}
}

if os.path.exists(validation_stats_file):
    with open(validation_stats_file, 'r') as f:
        try:
            validation_stats = json.load(f)
        except json.JSONDecodeError:
            pass

def load_system_message(filepath="system_prompt.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

# Exponential backoff for API rate limits
@backoff.on_exception(backoff.expo, 
                      (boto3.exceptions.Boto3Error, 
                       Exception),
                      max_tries=8,
                      base=2,
                      factor=3)
def call_bedrock_api(prompt, index):
    """Call Bedrock API with exponential backoff for rate limits"""
    system_message = load_system_message()
    
    # Fixed API request without response_format which isn't supported in Bedrock
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,  # Moderate temperature with validation step in place
        "system": system_message,
        "messages": [
            {"role": "user", "content": f"Analyze this prompt: {prompt}"}
        ]
    })
    
    try:
        response = bedrock.invoke_model(
            modelId="us.anthropic.claude-3-5-haiku-20241022-v1:0",  # Using Claude 3.5 Haiku
            body=body
        )
        
        response_body = json.loads(response.get('body').read())
        response_text = response_body['content'][0]['text']
        
        # Since we removed response_format, we need to extract the JSON from the response
        # Claude will sometimes add explanation before or after the JSON
        try:
            # Try to parse the entire response as JSON first
            parsed_json = json.loads(response_text)
            return parsed_json
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
                    
            # If that fails too, try to find json block with curly braces
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # Last resort fallback
                    print(f"Failed to parse JSON at index {index}, using fallback")
                    return {
                        "parallelizable": False,
                        "category": None,
                        "is_novel_category": False,
                        "category_description": None,
                        "serial": prompt,
                        "template": None,
                        "context": None,
                        "data": None,
                        "n": None
                    }
            
            # If all parsing attempts fail
            print(f"Failed to extract JSON at index {index}, using fallback")
            return {
                "parallelizable": False,
                "category": None,
                "is_novel_category": False,
                "category_description": None,
                "serial": prompt,
                "template": None,
                "context": None,
                "data": None,
                "n": None
            }
            
    except Exception as e:
        print(f"API call failed for index {index}: {str(e)}")
        raise

def validate_parallelizable(result, prompt):
    """Second check to validate if a query is truly parallelizable"""
    global validation_stats
    
    # 1. If Claude already said it's not parallelizable, accept that
    if not result.get("parallelizable", False):
        result["validation_tier"] = "not_parallelizable"
        return result
    
    validation_stats["total_classified_as_parallelizable"] += 1
    category = result.get("category", "Unknown")
    
    # 2. Check schema integrity first
    schema_valid = True
    
    # Check mutual exclusivity of data and n
    if result.get("data") is not None and result.get("n") is not None:
        schema_valid = False
        print(f"Schema error: Both 'data' and 'n' fields populated for index {result.get('index')}")
    
    # Check data field format
    data = result.get("data")
    if data is not None:
        if not isinstance(data, list):
            schema_valid = False
            print(f"Schema error: 'data' is not a list for index {result.get('index')}")
        elif len(data) < 2:
            schema_valid = False
            print(f"Schema error: 'data' list contains fewer than 2 items for index {result.get('index')}")
        elif any(not isinstance(item, str) for item in data):
            schema_valid = False
            print(f"Schema error: 'data' contains non-string items for index {result.get('index')}")
    
    # Check n field format
    n = result.get("n")
    if n is not None:
        if not isinstance(n, (int, float)) or n < 2:
            schema_valid = False
            print(f"Schema error: 'n' is not a valid number > 1 for index {result.get('index')}")
    
    # Check template format
    template = result.get("template")
    if template is not None:
        if category == "Repeated Generation" and "{n}" in template:
            schema_valid = False
            print(f"Schema error: Repeated Generation template contains '{{n}}' for index {result.get('index')}")
        elif category != "Repeated Generation" and (not "{data}" in template and not "{context}" in template):
            schema_valid = False
            print(f"Schema error: Non-Repeated Generation template missing placeholders for index {result.get('index')}")
    
    # 3. If schema is invalid, fail validation immediately
    if not schema_valid:
        result["parallelizable"] = False
        result["category"] = None
        result["is_novel_category"] = False
        result["category_description"] = None
        result["template"] = None
        result["context"] = None
        result["data"] = None
        result["n"] = None
        result["validation_tier"] = "not_parallelizable"
        validation_stats["failed_validation"] += 1
        
        if category in validation_stats["categories_failed"]:
            validation_stats["categories_failed"][category] += 1
        else:
            validation_stats["categories_failed"][category] = 1
            
        return result
    
    # 4. Proceed with content-based validation
    prompt_lower = prompt.lower()
    
    # More comprehensive verb and object lists
    verbs = r'(generate|create|write|make|give|provide|list|show|tell|name|identify|find|spot|translate|correct|develop|produce|craft|prepare|construct|compile|analyze|evaluate|compare|contrast|offer|suggest|need|want|require|discuss)'
    
    objects = r'(examples?|stories?|variations?|ideas?|options?|questions?|sentences?|paragraphs?|items?|tasks?|entities?|keywords?|words?|phrases?|translations?|summaries?|reports?|reviews?|analyses?|cases?|scenarios?|characters?|profiles?|descriptions?|suggestions?|recommendations?)'
    
    # Check for explicit numeric requests with expanded patterns
    has_numeric_request = bool(re.search(fr'\b{verbs}?\s*\d+\s*{objects}\b', prompt_lower))
    
    # Check for explicit list markers (enhanced for multi-language support)
    has_list_markers = bool(re.search(r'\b(following|these|each of|all of|todos?|las?s?|die|der|das|les?|la|il|и|и|и)\s+(questions?|items?|prompts?|sentences?|paragraphs?|tasks?|texts?|statements?|passages?|preguntas?|frases?|fragen|sätze|questions|phrases|вопросы|предложения)\b', prompt_lower))
    
    # Check for multiple numbered items in the prompt (enhanced pattern)
    has_numbered_items = len(re.findall(r'(?:\d+[\.\)\:]|(?:\n[-•*]\s+))', prompt)) > 1
    
    # Check for multiple questions in sequence (enhanced pattern)
    has_multiple_questions = len(re.findall(r'\?[\s\n]+', prompt)) > 1
    
    # Check for list format with bullets or dashes
    has_bullet_list = len(re.findall(r'(\n\s*[-•*]\s+|\n\s*\d+[\.\)]\s+)', prompt)) > 1
    
    # Check for keywords like "multiple" or "each" (enhanced for multi-language)
    has_multiplicity_markers = bool(re.search(r'\b(multiple|several|each|all|every|various|respectively|varios?s?|plusieurs|mehrere|多个|многие)\b', prompt_lower))
    
    # NEW: Check for comma-separated lists (at least 3 items)
    has_comma_list = bool(re.search(r'\b\w+\b\s*,\s*\b\w+\b\s*(?:,\s*(?:and\s+)?\b\w+\b)+', prompt_lower))
    
    # NEW: Check for "for each" pattern
    has_for_each_pattern = bool(re.search(r'\bfor\s+(every|each)\b', prompt_lower))
    
    # NEW: Check for plural nouns followed by list
    has_plural_list = bool(re.search(r'\b\w+s\b\s*[:;]\s*\b', prompt_lower))
    
    # Check if data/n looks good
    data_array = result.get("data", [])
    has_parallel_data = isinstance(data_array, list) and len(data_array) > 1
    
    n_value = result.get("n")
    has_multiple_n = isinstance(n_value, (int, float)) and n_value > 1
    
    # High confidence validation logic - strict criteria
    high_confidence = (
        has_numeric_request or 
        has_list_markers or 
        has_numbered_items or 
        has_bullet_list or
        has_multiple_questions or
        (has_multiplicity_markers and (has_parallel_data or has_multiple_n))
    )
    
    # NEW: Medium confidence validation - more permissive criteria
    medium_confidence = (
        has_comma_list or 
        has_for_each_pattern or
        has_plural_list or
        category in ["Reading Comprehension", "Named Entity Recognition", "Translation"] or
        (has_parallel_data and len(data_array) >= 3)
    )
    
    # Determine validation tier
    if high_confidence:
        validation_tier = "high_confidence"
        validation_stats["passed_validation"] += 1
        if category in validation_stats["categories_passed"]:
            validation_stats["categories_passed"][category] += 1
        else:
            validation_stats["categories_passed"][category] = 1
    elif medium_confidence:
        validation_tier = "medium_confidence"
        validation_stats["passed_validation"] += 1
        if category in validation_stats["categories_passed"]:
            validation_stats["categories_passed"][category] += 1
        else:
            validation_stats["categories_passed"][category] = 1
    else:
        validation_tier = "low_confidence"
        result["parallelizable"] = False
        result["category"] = None
        result["is_novel_category"] = False
        result["category_description"] = None
        result["template"] = None
        result["context"] = None
        result["data"] = None
        result["n"] = None
        validation_stats["failed_validation"] += 1
        
        if category in validation_stats["categories_failed"]:
            validation_stats["categories_failed"][category] += 1
        else:
            validation_stats["categories_failed"][category] = 1
    
    result["validation_tier"] = validation_tier
    
    # For backward compatibility
    result["validation_passed"] = (validation_tier in ["high_confidence", "medium_confidence"])
    
    return result

def is_parallelizable(prompt, index):
    """
    Use Claude to determine if a prompt is parallelizable.
    Returns a dict with the determination and explanation.
    """
    try:
        result = call_bedrock_api(prompt, index)

        result["index"] = index
        result["query_id"] = str(uuid.uuid4())[:8]
        result["prompt"] = prompt
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Apply validation and update shared stats
        result = validate_parallelizable(result, prompt)

        return result
    except Exception as e:
        print(f"Error analyzing prompt at index {index}: {e}")
        return {
            "index": index,
            "query_id": str(uuid.uuid4())[:8],
            "prompt": prompt,
            "parallelizable": False,
            "category": None,
            "is_novel_category": False,
            "category_description": None,
            "serial": prompt,
            "template": None,
            "context": None,
            "data": None,
            "n": None,
            "validation_tier": "not_parallelizable",
            "validation_passed": False,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
def save_to_csv(result):
    """Save a single result to CSV file"""
    # Only save parallelizable queries that passed validation
    if not result.get("parallelizable", False):
        return
        
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        # Filter to only include fields we want in our CSV
        row = {field: result.get(field, "") for field in field_names}
        
        # Convert lists to string representation for CSV
        if isinstance(row.get("data"), list):
            row["data"] = json.dumps(row["data"])
            
        writer.writerow(row)

def save_validation_stats():
    """Save validation statistics to file"""
    with open(validation_stats_file, 'w') as f:
        json.dump(validation_stats, f, indent=2)

def update_novel_categories(category, description, example):
    """Update the novel categories tracking file"""
    novel_categories = {}
    if os.path.exists(novel_categories_file):
        with open(novel_categories_file, 'r') as f:
            try:
                novel_categories = json.load(f)
            except json.JSONDecodeError:
                novel_categories = {}
    
    if category not in novel_categories:
        novel_categories[category] = {
            "description": description,
            "examples": [example],
            "count": 1
        }
    else:
        novel_categories[category]["count"] += 1
        if len(novel_categories[category]["examples"]) < 5:  # Store up to 5 examples
            novel_categories[category]["examples"].append(example)
    
    with open(novel_categories_file, 'w') as f:
        json.dump(novel_categories, f, indent=2)

def process_prompt(args):
    """Process a single prompt (for use with ThreadPoolExecutor)"""
    prompt, index, novel_categories = args
    
    # Skip processing if prompt is empty
    if not prompt or len(prompt) < 10:
        return None
    
    # Analyze the prompt
    result = is_parallelizable(prompt, index)
    
    # Save result immediately
    save_to_csv(result)
    
    # Track novel categories
    if result.get("parallelizable", False) and result.get("is_novel_category", False):
        cat = result.get("category")
        if cat and cat not in novel_categories:
            # Update novel categories file
            update_novel_categories(
                cat, 
                result.get("category_description", ""), 
                {"prompt": prompt[:500], "index": index}
            )
            print(f"\n!!! NEW CATEGORY DISCOVERED: {cat} !!!")
            print(f"Description: {result.get('category_description')}")
            print(f"Example prompt: {prompt[:100]}...")
    
    return result

def filter_prompts(batch):
    """Extract valid prompts from a batch of data"""
    valid_prompts = []
    for item in batch:
        # Extract the user prompt (first message in the conversation)
        messages = item.get("conversation", [])
        if not messages:
            continue
        
        # Get the first user message
        prompt = None
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "").strip()
                break
        
        if prompt and len(prompt) >= 10:  # Skip empty or very short prompts
            valid_prompts.append(prompt)
        else:
            valid_prompts.append(None)
    
    return valid_prompts

def main():
    # Get total dataset size
    total_size = len(dataset["train"])
    print(f"Total dataset size: {total_size} entries")
    
    # Start position (useful for resuming)
    start_position = 0
    
    # Check if we need to resume from a previous run
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        if not df.empty:
            # Get the highest index we've processed so far
            start_position = df['index'].max() + 1
            print(f"Resuming from position {start_position}")
    
    # Process in batches to enable easier resuming
    batch_size = 100  # Reduced batch size
    max_workers = 3  # Reduced workers for Haiku which may have rate limits
    
    # Stats tracking
    stats = {
        "processed": 0,
        "parallelizable": 0,
        "categories": {},
        "novel_categories": {}
    }
    
    try:
        # Process the dataset in batches
        for batch_start in tqdm(range(start_position, total_size, batch_size)):
            batch_end = min(batch_start + batch_size, total_size)
            print(f"\nProcessing batch {batch_start} to {batch_end-1}")
            
            # Select the current batch
            batch_indices = range(batch_start, batch_end)
            batch = dataset["train"].select(batch_indices)
            
            # Filter and get valid prompts
            prompts = filter_prompts(batch)
            
            # Process prompts in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create tasks with indices
                tasks = [(prompt, batch_start + i, novel_categories_lock) for i, prompt in enumerate(prompts) if prompt]
                
                # Submit all tasks
                futures = [executor.submit(process_prompt, task) for task in tasks]
                
                # Process results as they complete
                for future in tqdm(futures, desc="Processing prompts"):
                    try:
                        result = future.result(timeout=90)  # Longer timeout
                        if result:
                            stats["processed"] += 1
                            if result.get("parallelizable", False):
                                stats["parallelizable"] += 1
                                cat = result.get("category")
                                is_novel = result.get("is_novel_category", False)
                                
                                if is_novel:
                                    if cat in stats["novel_categories"]:
                                        stats["novel_categories"][cat] += 1
                                    else:
                                        stats["novel_categories"][cat] = 1
                                else:
                                    if cat in stats["categories"]:
                                        stats["categories"][cat] += 1
                                    else:
                                        stats["categories"][cat] = 1
                    except Exception as e:
                        print(f"Task failed: {e}")
            
            # Print stats after each batch
            print(f"\nStats after {batch_end} prompts:")
            if stats["processed"] > 0:  # Avoid division by zero
                print(f"Parallelizable: {stats['parallelizable']} ({stats['parallelizable']/stats['processed']*100:.2f}% of processed)")
            print("Known Categories:")
            for cat, count in stats["categories"].items():
                print(f"- {cat}: {count}")
            if stats["novel_categories"]:
                print("Novel Categories:")
                for cat, count in stats["novel_categories"].items():
                    print(f"- {cat}: {count}")
            
            # Print validation stats
            print("\nValidation Stats:")
            print(f"Total classified as parallelizable: {validation_stats['total_classified_as_parallelizable']}")
            print(f"Passed validation: {validation_stats['passed_validation']} ({validation_stats['passed_validation']/max(1, validation_stats['total_classified_as_parallelizable'])*100:.2f}%)")
            print(f"Failed validation: {validation_stats['failed_validation']} ({validation_stats['failed_validation']/max(1, validation_stats['total_classified_as_parallelizable'])*100:.2f}%)")
            
            # Save validation stats periodically
            save_validation_stats()
                    
            # Add a delay between batches to avoid rate limiting
            time.sleep(7)  # Short delay for Haiku
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
    except Exception as e:
        print(f"\nError encountered: {e}")
    finally:
        print("\nFinal stats:")
        print(f"Processed: {stats['processed']}")
        print(f"Parallelizable: {stats['parallelizable']}")
        print("Known Categories:")
        for cat, count in stats["categories"].items():
            print(f"- {cat}: {count}")
        if stats["novel_categories"]:
            print("Novel Categories:")
            for cat, count in stats["novel_categories"].items():
                print(f"- {cat}: {count}")
        
        # Save final validation stats
        save_validation_stats()


if __name__ == "__main__":
    main()