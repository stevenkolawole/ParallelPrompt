import json
import random
import argparse
import os
from pydantic import BaseModel
from openai import OpenAI
import tqdm


class LLMJudgeResponse(BaseModel):
    """
    Pydantic model for parsing the LLM judge's response.
    """
    accuracy: int  # 1 for response 1, 2 for response 2, 0 if tied
    grammar: int   # 1 for response 1, 2 for response 2, 0 if tied
    detail: int    # 1 for response 1, 2 for response 2, 0 if tied
    preference: int  # 1 for response 1, 2 for response 2, 0 if tied
    reasoning: str    # Explanation for the scores


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate LLM responses using an LLM as a judge."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSON file containing prompts and responses.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to the output JSON file to save evaluation results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="OpenAI model to use for evaluation (default: gpt-4).",
    )
    args = parser.parse_args()

    # Initialize OpenAI client
    client = OpenAI()

    # Define the system prompt and initial user message
    messages = [
        {
            "role": "system",
            "content": """
            You are an impartial judge tasked with comparing two LLM responses. You will answer four specific questions based on the two responses provided, using the following criteria:

            1. **Accuracy**: Which response more accurately follows the instructions given in the prompt?
               - Score 1 if response 1 is more accurate, 2 if response 2 is more accurate, or 0 if they are equally accurate or equally inaccurate.

            2. **Grammar**: Which response is more grammatically correct?
               - Score 1 if response 1 is grammatically superior, 2 if response 2 is better, or 0 if they are equally grammatically correct (or incorrect).

            3. **Detail**: Which response provides more detail and specificity?
               - Score 1 if response 1 is more detailed, 2 if response 2 is more detailed, or 0 if they are equally detailed (or equally lacking in detail).

            4. **Preference**: Which response do you personally prefer overall, considering all factors?
               - Score 1 if you prefer response 1 overall, 2 if you prefer response 2, or 0 if you are equally satisfied with both responses.

            5. **Reasoning**: Explain your reasoning for your above answers.
                - Provide a short paragraph explaining your reasoning for the above questions.

            The format of your response must consist of a JSON object with the following structure:
            {
                "accuracy": 1 | 2 | 0,
                "grammar": 1 | 2 | 0,
                "detail": 1 | 2 | 0,
                "preference": 1 | 2 | 0,
                "reasoning": str,
            }
            """,
        },
        {
            "role": "user",
            "content": "Here is the prompt and two responses. Compare them based on accuracy, grammar, detail, and preference, and explain your reasoning.",
        },
    ]

    # Load the data from the input file
    if not os.path.exists(args.input):
        print(f"Input file '{args.input}' not found.")
        return

    with open(args.input, "r") as f:
        data = json.load(f)

    # Initialize the results list
    results = []

    # Iterate over each request in the data
    for request in tqdm.tqdm(data):
        if "prompt" not in request:
            continue

        prompt = request["prompt"]
        response_1 = request["serial_output"]  # The "serial" response
        response_2 = "\n\n".join(request["parallel_output"])  # The "parallel" response

        # Shuffle the responses to avoid bias
        responses = [response_1, response_2]
        original_order = ["serial", "parallel"]
        shuffled_order = list(zip(responses, original_order))
        random.shuffle(shuffled_order)

        # Unpack shuffled responses and labels
        shuffled_responses = [resp for resp, _ in shuffled_order]
        shuffled_labels = [label for _, label in shuffled_order]

        # Construct the messages to send to the API
        conversation = messages + [
            {"role": "user", "content": f"Prompt: {prompt}"},
            {"role": "user", "content": f"Response 1: {shuffled_responses[0]}"},
            {"role": "user", "content": f"Response 2: {shuffled_responses[1]}"},
        ]

        # Send the prompt and shuffled responses to the API for evaluation
        try:
            completion = client.beta.chat.completions.parse(
                model=args.model,
                response_format=LLMJudgeResponse,
                messages=conversation,
            )
        except Exception as e:
            print(f"Error during API call: {e}")
            continue

        # Capture the evaluation from the response
        message = completion.choices[0].message
        if message.parsed:
            # Extract evaluation scores
            evaluation = {
                "accuracy": message.parsed.accuracy,
                "grammar": message.parsed.grammar,
                "detail": message.parsed.detail,
                "preference": message.parsed.preference,
                "reasoning": message.parsed.reasoning,
            }

            # Re-map the scores to the original order (serial vs parallel)
            mapped_evaluation = {}
            for key in ['accuracy', 'grammar', 'detail', 'preference']:
                score = evaluation[key]
                if score == 0:
                    mapped_score = 0  # Tie
                else:
                    # Map back to original labels
                    winner_label = shuffled_labels[score - 1]
                    mapped_score = 1 if winner_label == "serial" else 2
                mapped_evaluation[key] = mapped_score

            # Include reasoning in the mapped evaluation
            mapped_evaluation['reasoning'] = evaluation['reasoning']

            # Store the result with the prompt and evaluation
            results.append(
                {
                    "prompt": prompt,
                    "response_1": response_1,  # Always "serial"
                    "response_2": response_2,  # Always "parallel"
                    "evaluation": mapped_evaluation,
                }
            )
        else:
            print(f"Parsing error or refusal: {message.refusal}")

    # Save the results to the output JSON file
    with open(args.output, "w") as outfile:
        json.dump(results, outfile, indent=4)
    print(f"Evaluation results saved to '{args.output}'.")


if __name__ == "__main__":
    main()
