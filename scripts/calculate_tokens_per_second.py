import json
import os

# Path to the JSONL file
jsonl_file_path = "data/meta-llama/Llama-3.2-1B-Instruct/beam_search_n4_temp1.0_exp_20251210_141452_completions.jsonl"

# Check if the file exists
if not os.path.exists(jsonl_file_path):
    print(f"File not found: {jsonl_file_path}")
    exit(1)

# Load the JSONL file and calculate token/s
with open(jsonl_file_path, 'r') as file:
    total_completion_count = 0
    total_completions = 0

    for line_number, line in enumerate(file, start=1):
        try:
            # Parse the JSON line
            data = json.loads(line)

            # Extract the required fields
            llm_gen_time = data.get("llm_gen_time", 0)  # Time for LLM generation
            prm_score_time = data.get("prm_score_time", 0)  # Time for PRM scoring
            completion_tokens = data.get("completion_tokens", [])
            completions = data.get("completions", [])

            # Calculate token count and tokens/s
            if completion_tokens:
                token_count = completion_tokens[0]  # First element in completion_tokens

                # Calculate LLM generation tokens/s
                if llm_gen_time > 0:
                    llm_tokens_per_second = token_count / llm_gen_time
                    print(f"Line {line_number}: LLM Gen - {llm_tokens_per_second:.2f} tokens/s")
                else:
                    print(f"Line {line_number}: LLM Gen time is zero or invalid")

                # Calculate PRM scoring tokens/s
                if prm_score_time > 0:
                    prm_tokens_per_second = token_count / prm_score_time
                    print(f"Line {line_number}: PRM Score - {prm_tokens_per_second:.2f} tokens/s")
                else:
                    print(f"Line {line_number}: PRM Score time is zero or invalid")
            else:
                print(f"Line {line_number}: No tokens in completion_tokens")

            # Calculate total number of elements in completions
            total_completion_count += len(completions)
            total_completions += 1

        except json.JSONDecodeError as e:
            print(f"Line {line_number}: Failed to parse JSON - {e}")
        except Exception as e:
            print(f"Line {line_number}: Error - {e}")

    # Calculate and print average number of elements in completions
    if total_completions > 0:
        average_completion_count = total_completion_count / total_completions
        print(f"\nAverage number of elements in completions: {average_completion_count:.2f}")
    else:
        print("\nNo completions found to calculate average number of elements.")

'''
(base) [yf3005@ga007 search-and-learn]$ python scripts/calculate_tokens_per_second.py\
> 
Line 1: LLM Gen - 167.22 tokens/s
Line 1: PRM Score - 437.62 tokens/s
Line 2: LLM Gen - 56.49 tokens/s
Line 2: PRM Score - 76.03 tokens/s
Line 3: LLM Gen - 199.34 tokens/s
Line 3: PRM Score - 280.00 tokens/s
Line 4: LLM Gen - 172.63 tokens/s
Line 4: PRM Score - 509.62 tokens/s

Average number of elements in completions: 4.00
'''