import os
import json
from collections import OrderedDict
import tqdm
import openai


def convert_to_data_parallel(
    input_file, base_prompt_file, output_file, tools, order_keys_func, task_limit=None
):
    """
    Process tasks by converting prompts to data parallel tasks using the OpenAI API.

    Parameters:
    - input_file (str): Path to the file containing the original prompts.
    - base_prompt_file (str): Path to the file containing the base prompt.
    - output_file (str): Path to the JSON file where results will be stored.
    - tools (list): List of tools (functions) to pass to the OpenAI API.
    - order_keys_func (callable): Function to order the keys in the task dictionary.
    - task_limit (int, optional): Limit on the number of tasks to process. Defaults to None.

    Returns:
    - None
    """
    # Read the file content
    with open(input_file, "r") as f:
        tasks = f.readlines()

    # Apply task limit if provided
    if task_limit is not None:
        tasks = tasks[:task_limit]

    # Read the base prompt
    with open(base_prompt_file, "r") as f:
        base_prompt = f.read()

    # Initialize OpenAI client
    client = openai.OpenAI()

    # Read existing results if output file exists
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Create a set of prompts already processed
    completed_prompts = set(x["original"] for x in results)

    # Process each task
    for x in tqdm.tqdm(tasks):
        x = x.strip()
        if x in completed_prompts:
            continue

        # Build the prompt
        prompt = base_prompt + f'\noriginal_prompt = """{x}"""'

        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that converts language model prompts to data parallel tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": "convert_to_data_parallel"},
                },
            )

            # Extract the task from the response
            task = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            task["original"] = x

            # Order the keys in the task dictionary
            task = order_keys_func(task)

            # Add the task to results
            results.append(task)

            # Write the updated results to the output file
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            print(f"An error occurred while processing prompt: {x}\nError: {e}")
            continue
