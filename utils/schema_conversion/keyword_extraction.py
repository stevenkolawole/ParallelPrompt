# keyword_extraction.py

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
            ("data", task.get("data")),
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
                            "description": "Any relevant context information to include in the data parallel template.",
                            "type": "string",
                        },
                        "data": {
                            "description": "The list of data parallel items to instantiate the template with.",
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["serial", "template", "context", "data"],
                },
            },
        }
    ]

    convert_to_data_parallel(
        input_file="keyword_extraction.txt",
        base_prompt_file="keyword_extraction_base_prompt.txt",
        output_file="keyword_extraction_lmsys.json",
        tools=tools,
        order_keys_func=order_keys,
    )
