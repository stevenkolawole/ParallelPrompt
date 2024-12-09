import json
from collections import OrderedDict
from convert_to_data_parallel import convert_to_data_parallel


def order_keys(task):
    """
    Orders the keys in the task dictionary for consistent output.

    Parameters:
    - task (dict): Dictionary containing task data.

    Returns:
    - OrderedDict: Ordered dictionary with keys in specified order.
    """
    return OrderedDict(
        [
            ("original", task["original"]),
            ("serial", task["serial"]),
            ("template", task.get("template")),
            ("context", task.get("context")),
            ("n", task.get("n")),
        ]
    )


if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "convert_to_data_parallel",
                "description": "Converts a language model prompt to a data parallel task represented as a JSON object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "serial": {
                            "description": "A cleaned-up version of the prompt string that is meant to be executed serially.",
                            "type": "string",
                        },
                        "template": {
                            "description": "A template for data parallel generation which may include some context.",
                            "type": "string",
                        },
                        "context": {
                            "description": "Any relevant context information to include in the data parallel template, if necessary.",
                            "type": "string",
                        },
                        "n": {
                            "description": "The number of times to invoke the task.",
                            "type": "integer",
                        },
                    },
                    "required": ["serial", "template", "context", "n"],
                },
            },
        }
    ]

    convert_to_data_parallel(
        input_file="generate_n.txt",
        base_prompt_file="generate_n_base_prompt.txt",
        output_file="generate_n_lmsys.json",
        tools=tools,
        order_keys_func=order_keys,
        task_limit=120,
    )
