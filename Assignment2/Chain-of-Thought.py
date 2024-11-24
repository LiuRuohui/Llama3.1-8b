import time
import os
from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from evaluate_functional_correctness import entry_point
from openai import OpenAI
import openai

# API Keys List
API_KEYS = [
    "bab5a926-5245-4843-a03d-d98b57a0c644",
    "31a55a95-3f5c-483b-9a35-5fa473a6006a",
    "1fe618ed-bb32-408e-a481-57f9c477692f",
    "60c63e5e-0082-4fa6-a642-f8d678f61ddc",
    "f1d22e5a-d33e-4d1a-a3f9-c99e22749b0b"
]
api_key_index = 0

client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

FEW_SHOT_EXAMPLES_SCOT = [
    {
        "input": (
            "def first_Repeated_Char(str):\n    "
            "```Write a python function to find the first repeated character in a given string.\n    ```\n pass\n "
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
            ),
        "output": (
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1:    for each character in str:\n"
            "2:      if ch appears more than once in str:\n"
            "3:        return ch\n"
            "4:    return None\n"
        )
    },
    {
    "input": (
        "def max_Subarray_Sum(arr):\n    "
        "```Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n    ```\n pass\n"
        "Please understand the requirement and write a rough solving process."
        "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
        " The necessary details should be written in natural languages."
    ),
    "output": (
        "Input: arr: a list of integers\n"
        "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
        "1:    Initialize max_sum and current_sum with the first element of arr\n"
        "2:    for each subsequent element in arr:\n"
        "3:        update current_sum as the larger value between the current element and (current_sum + current element)\n"
        "4:        update max_sum as the larger value between max_sum and current_sum\n"
        "5:    return max_sum\n"
    )
    },
    {
    "input": (
        "def word_Frequency(text):\n    "
        "```Write a Python function to count the frequency of each word in a given string.\n    ```\n pass\n"
        "Please understand the requirement and write a rough solving process."
        "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
        " The necessary details should be written in natural languages."
    ),
    "output": (
        "Input: text: a string containing words separated by spaces\n"
        "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
        "1:    Split the text into words based on spaces\n"
        "2:    Initialize an empty dictionary freq_dict\n"
        "3:    for each word in the list of words:\n"
        "4:        if the word is already in freq_dict, increment its count\n"
        "5:        otherwise, add the word to freq_dict with a count of 1\n"
        "6:    return freq_dict\n"
    )
    }
]

FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "def first_Repeated_Char(str):\n    "
            "```Write a python function to find the first repeated character in a given string.\n"
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1:    for each character in str:\n"
            "2:      if ch appears more than once in str:\n"
            "3:        return ch\n"
            "4:    return None ```\n"
            "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
            ),
        "output": (
            "def first_Repeated_Char(str):\n"
            "    h = {}\n"
            "    for ch in str:\n"
            "        if ch in h:\n"
            "            return ch;\n"
            "        else:\n"
            "            h[ch] = 0\n"
            "    return None\n"
        )
    },
    {
    "input": (
        "def max_Subarray_Sum(arr):\n    "
        "```Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n"
        "Input: arr: a list of integers\n"
        "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
        "1:    Initialize max_sum and current_sum with the first element of arr\n"
        "2:    for each subsequent element in arr:\n"
        "3:        update current_sum as the larger value between the current element and (current_sum + current element)\n"
        "4:        update max_sum as the larger value between max_sum and current_sum\n"
        "5:    return max_sum ```\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def max_Subarray_Sum(arr):\n"
        "    max_sum = arr[0]\n"
        "    current_sum = arr[0]\n"
        "    for num in arr[1:]:\n"
        "        current_sum = max(num, current_sum + num)\n"
        "        max_sum = max(max_sum, current_sum)\n"
        "    return max_sum\n"
    )
    },
    {
    "input": (
        "def word_Frequency(text):\n    "
        "```Write a Python function to count the frequency of each word in a given string.\n"
        "Input: text: a string containing words separated by spaces\n"
        "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
        "1:    Split the text into words based on spaces\n"
        "2:    Initialize an empty dictionary freq_dict\n"
        "3:    for each word in the list of words:\n"
        "4:        if the word is already in freq_dict, increment its count\n"
        "5:        otherwise, add the word to freq_dict with a count of 1\n"
        "6:    return freq_dict ```\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def word_Frequency(text):\n"
        "    freq_dict = {}\n"
        "    words = text.split()\n"
        "    for word in words:\n"
        "        if word in freq_dict:\n"
        "            freq_dict[word] += 1\n"
        "        else:\n"
        "            freq_dict[word] = 1\n"
        "    return freq_dict\n"
    )
    }
]

"""
This prompt used for demonstration wrong
FEW_SHOT_EXAMPLES_SCOT_MODIFIED = [
    {
        "input": (
            "def first_Repeated_Char(str):\n "
            "Write a python function to find the first repeated character in a given string.\n \n pass\n "
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1: for each character in str:\n"
            "2: if ch appears more than once in str:\n"
            "3:if ch == 'a':\n"
            "4:return ch\n"
            "5: return None\n"
        )
    },
    {
        "input": (
            "def max_Subarray_Sum(arr):\n "
            "Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n \n pass\n"
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: arr: a list of integers\n"
            "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
            "1: Initialize max_sum and current_sum with the first element of arr\n"
            "2: for each subsequent element in arr:\n"
            "3:update current_sum as current_sum + arr[0]\n"
            "4: update max_sum as the larger value between max_sum and current_sum\n"
            "5: return max_sum\n"
        )
    },
    {
        "input": (
            "def word_Frequency(text):\n "
            "Write a Python function to count the frequency of each word in a given string.\n \n pass\n"
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: text: a string containing words separated by spaces\n"
            "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
            "1: Split the text into words based on spaces\n"
            "2: Initialize an empty dictionary freq_dict\n"
            "3: for each word in the list of words:\n"
            "4:if len(word) > 5:\n"
            "5:freq_dict[word] = 1\n"
            "6: return freq_dict\n"
        )
    }
    ]

FEW_SHOT_EXAMPLES_MODIFIED = [
    {
        "input": (
            "def first_Repeated_Char(str):\n    "
            "```Write a python function to find the first repeated character in a given string.\n"
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1: for each character in str:\n"
            "2: if ch appears more than once in str:\n"
            "3:if ch == 'a':\n"
            "4:return ch\n"
            "5: return None\n"
            "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
            ),
        "output": (
            "def first_Repeated_Char(str):\n"
            "    h = {}\n"
            "    for ch in str:\n"
            "        if ch in h:\n"
            "            return ch;\n"
            "        else:\n"
            "            h[ch] = 1\n"
            "    return None\n"
        )
    },
    {
    "input": (
        "def max_Subarray_Sum(arr):\n    "
        "```Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n"
            "Input: arr: a list of integers\n"
            "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
            "1: Initialize max_sum and current_sum with the first element of arr\n"
            "2: for each subsequent element in arr:\n"
            "3:update current_sum as current_sum + arr[0]\n"
            "4: update max_sum as the larger value between max_sum and current_sum\n"
            "5: return max_sum\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def max_Subarray_Sum(arr):\n"
        "    max_sum = arr[0]\n"
        "    current_sum = arr[0]\n"
        "    for num in arr[1:]:\n"
        "        current_sum = num - current_sum\n"
        "        max_sum = max(max_sum, current_sum)\n"
        "    return max_sum\n"
    )
    },
    {
    "input": (
        "def word_Frequency(text):\n    "
        "```Write a Python function to count the frequency of each word in a given string.\n"
            "Input: text: a string containing words separated by spaces\n"
            "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
            "1: Split the text into words based on spaces\n"
            "2: Initialize an empty dictionary freq_dict\n"
            "3: for each word in the list of words:\n"
            "4:if len(word) > 5:\n"
            "5:freq_dict[word] = 1\n"
            "6: return freq_dict\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def word_Frequency(text):\n"
        "    freq_dict = {}\n"
        "    words = text.split()\n"
        "    for word in words:\n"
        "        if word not in freq_dict:\n"
        "            freq_dict[word] += 1\n"
        "        else:\n"
        "            freq_dict[word] = 1\n"
        "    return freq_dict\n"
    )
    }
]
"""

"""
This prompt used for demonstration irrelevant
FEW_SHOT_EXAMPLES_SCOT_MODIFIED = [
    {
        "input": (
            "def first_Repeated_Char(str):\n "
            "Write a python function to find the first repeated character in a given string.\n \n pass\n "
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1: for each character in str:\n"
            "2: if ch appears more than once in str:\n"
            "3:if ch == 'a':\n"
            "4:return ch\n"
            "5: return None\n"
        )
    },
    {
        "input": (
            "def max_Subarray_Sum(arr):\n "
            "Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n \n pass\n"
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: arr: a list of integers\n"
            "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
            "1: Initialize max_sum and current_sum with the first element of arr\n"
            "2: for each subsequent element in arr:\n"
            "3:update current_sum as current_sum + arr[0]\n"
            "4: update max_sum as the larger value between max_sum and current_sum\n"
            "5: return max_sum\n"
        )
    },
    {
        "input": (
            "def word_Frequency(text):\n "
            "Write a Python function to count the frequency of each word in a given string.\n \n pass\n"
            "Please understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."
        ),
        "output": (
            "Input: text: a string containing words separated by spaces\n"
            "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
            "1: Split the text into words based on spaces\n"
            "2: Initialize an empty dictionary freq_dict\n"
            "3: for each word in the list of words:\n"
            "4:if len(word) > 5:\n"
            "5:freq_dict[word] = 1\n"
            "6: return freq_dict\n"
        )
    }
    ]

FEW_SHOT_EXAMPLES_MODIFIED = [
    {
        "input": (
            "def first_Repeated_Char(str):\n    "
            "```Write a python function to find the first repeated character in a given string.\n"
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1: for each character in str:\n"
            "2: if ch appears more than once in str:\n"
            "3:if ch == 'a':\n"
            "4:return ch\n"
            "5: return None\n"
            "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
            ),
        "output": (
            "def first_Repeated_Char(str):\n"
            "    h = {}\n"
            "    for ch in str:\n"
            "        if ch in h:\n"
            "            return ch;\n"
            "        else:\n"
            "            h[ch] = 1\n"
            "    return None\n"
        )
    },
    {
    "input": (
        "def max_Subarray_Sum(arr):\n    "
        "```Write a Python function to find the maximum sum of a contiguous subarray in a given array.\n"
            "Input: arr: a list of integers\n"
            "Output: max_sum: the maximum sum of any contiguous subarray in arr\n"
            "1: Initialize max_sum and current_sum with the first element of arr\n"
            "2: for each subsequent element in arr:\n"
            "3:update current_sum as current_sum + arr[0]\n"
            "4: update max_sum as the larger value between max_sum and current_sum\n"
            "5: return max_sum\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def max_Subarray_Sum(arr):\n"
        "    max_sum = arr[0]\n"
        "    current_sum = arr[0]\n"
        "    for num in arr[1:]:\n"
        "        current_sum = num - current_sum\n"
        "        max_sum = max(max_sum, current_sum)\n"
        "    return max_sum\n"
    )
    },
    {
    "input": (
        "def word_Frequency(text):\n    "
        "```Write a Python function to count the frequency of each word in a given string.\n"
            "Input: text: a string containing words separated by spaces\n"
            "Output: freq_dict: a dictionary where keys are words and values are their counts\n"
            "1: Split the text into words based on spaces\n"
            "2: Initialize an empty dictionary freq_dict\n"
            "3: for each word in the list of words:\n"
            "4:if len(word) > 5:\n"
            "5:freq_dict[word] = 1\n"
            "6: return freq_dict\n"
        "#Please check the above solving process and write a code base on it. Note that the solving process may contain errors.\n"
    ),
    "output": (
        "def word_Frequency(text):\n"
        "    freq_dict = {}\n"
        "    words = text.split()\n"
        "    for word in words:\n"
        "        if word not in freq_dict:\n"
        "            freq_dict[word] += 1\n"
        "        else:\n"
        "            freq_dict[word] = 1\n"
        "    return freq_dict\n"
    )
    }
]
"""
def question_prompt(s):
    return f'Question: {s}'

def construct_few_shot_chats_SCOT(n):
    """Construct few - shot chats based on the FEW_SHOT_EXAMPLES."""
    chats = []
    #for example in FEW_SHOT_EXAMPLES_SCOT[:n]:
    for example in FEW_SHOT_EXAMPLES_SCOT_MODIFIED[:n]:
        chats.append({"role": "user", "content": question_prompt(example["input"])})
        chats.append({"role": "assistant", "content": example["output"]})
    return chats

def construct_few_shot_chats(n):
    """Construct few - shot chats based on the FEW_SHOT_EXAMPLES."""
    chats = []
    #for example in FEW_SHOT_EXAMPLES[:n]:
    for example in FEW_SHOT_EXAMPLES_MODIFIED[:n]:
        chats.append({"role": "user", "content": question_prompt(example["input"])})
        chats.append({"role": "assistant", "content": example["output"]})
    return chats

def clean_the_wrap(code):
    # Remove ``` markers and the word 'python'
    start_index = code.find('```')
    if start_index!= -1:
        end_index = code.find('```', start_index + 3)
        if end_index!= -1:
            code = code[start_index + 3:end_index].replace("python", "")
    return code.strip()

def extract_first_function(prompt):
    lines = prompt.split("\n")
    for line in lines:
        if "def" in line:
            return line.strip()
    return None

def CoT_generation(prompt, cot_enabled=False, scot_enabled=False, scot_fewshot_enabled=False, retries=100, delay=1):
    global api_key_index

    if cot_enabled:
        # add CoT prompt
        cot_prompt = (
            "Let's solve this problem step by step. Begin by analyzing the problem requirements. "
            "Then break it down into smaller components. After that, implement a function that addresses the problem. "
        )
        chats = f"{cot_prompt}\n\n{prompt}"

    if scot_enabled:
        SCoT_prompt = (
            "Please understand the requirement and write a rough solving process.\n"
            "It starts with a input - output structure.\n"
            "You should use three basic structures to build the solving process, including sequences, branches, and loops.\n"
            "The necessary details should be written in natural languages.\n\n"
        )
        #prompt = f"\n{prompt}\n Please understand the requirement and write a rough solving process.\n You should use three basic structures to build the solving process, including sequences, branches, and loops.\n The necessary details should be written in natural languages.\n Do not change the function name given.\n"
        chats = f"\n{prompt}\n{SCoT_prompt}\n."

    if scot_fewshot_enabled:
        fewshot_chats = construct_few_shot_chats_SCOT(n=3)
        new_chat = [{"role": "user", "content": question_prompt(prompt)+"\nPlease understand the requirement and write a rough solving process."
            "It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops."
            " The necessary details should be written in natural languages."}]
        chats = fewshot_chats + new_chat

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=chats,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens = 1024
            )

            full_response = ""
            tokens = 0
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        full_response += delta.content
                if chunk.usage:
                    tokens += chunk.usage.completion_tokens

            return full_response, tokens, f"{chats}"  # return full response and tokens and the prompt given

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0
def code_generation(CoT, prompt, cot_enabled=False, scot_enabled=False, scot_fewshot_enabled=False, retries=100, delay=1):
    global api_key_index
    if cot_enabled:
        # add CoT prompt
        chats = f"{CoT}"

    if scot_enabled:
        chats = f"\n{CoT}\n# Please check the above solving process and write a code based on it. Note that the solving process may contain errors.\n"

    if scot_fewshot_enabled:
        fewshot_chats = construct_few_shot_chats(n=3)
        new_chat = [{"role": "user", "content": question_prompt(CoT)+ f"# Please check the above solving process and write a code based on it. Note that the solving process may contain errors.\n The function name should be {extract_first_function(prompt)}"}]
        chats = fewshot_chats + new_chat

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=chats,
                stream=True,
                stream_options={"include_usage": True},
                max_tokens = 2048
            )
            full_response = ""
            tokens = 0
            for chunk in completion:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content'):
                        full_response += delta.content
                if chunk.usage:
                    tokens += chunk.usage.completion_tokens

            return full_response, tokens, chats

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

if __name__ == '__main__':

    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        #CoT baseline
        #completion, token_count, prompt_given = CoT_generation(prompt, cot_enabled=True)
        #SCoT Improvement
        #completion, token_count1, prompt_given = CoT_generation(prompt, scot_enabled=True)
        completion, token_count1, prompt_given = CoT_generation(prompt, scot_fewshot_enabled=True)
        total_tokens += token_count1
        #print("model output for SCoT\n",completion)
        #completion, token_count2, prompt_given = code_generation(completion, prompt, cot_enabled=True)
        #completion, token_count2, prompt_given = code_generation(completion, prompt, scot_enabled=True)
        completion, token_count2, prompt_given = code_generation(completion, prompt, scot_fewshot_enabled=True)
        #print("model output for code\n",completion)
        elapsed_time = time.time() - start_time
        #print("clean the wrap of the model output \n", clean_the_wrap(completion))

        total_time += elapsed_time  # accumulate time
        total_tokens += token_count2  # accumulate token

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": clean_the_wrap(completion),
                "elapsed_time": elapsed_time,
                "token_count": token_count1+token_count2
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count1+token_count2}")

    # average time and token
    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    #write_jsonl("cot_baseline.jsonl", generated_solutions)
    #write_jsonl("scot_baseline_zeroshot.jsonl", generated_solutions)
    #write_jsonl("scot_baseline_fewshot.jsonl", generated_solutions)
    write_jsonl("scot_baseline_fewshot_demowrong.jsonl", generated_solutions)
    #result = entry_point("cot_baseline.jsonl", k="1", n_workers=4, timeout=5.0)
    #result = entry_point("scot_baseline_zeroshot.jsonl", k="1", n_workers=4, timeout=5.0)
    #result = entry_point("scot_baseline_fewshot.jsonl", k="1", n_workers=4, timeout=5.0)
    result = entry_point("scot_baseline_fewshot_demowrong.jsonl", k="1", n_workers=4, timeout=5.0)

    """
    # Compare
    passed_task_ids_cot = []
    passed_task_ids_scot = []
    passed_task_ids_scot_few = []
    results_cot = read_problems("cot_baseline.jsonl_results.jsonl")
    results_scot = read_problems("scot_baseline_zeroshot.jsonl_results.jsonl")
    results_scot_few = read_problems("scot_baseline_fewshot.jsonl_results.jsonl")
    for r in results_cot:
        if results_cot[r]["result"] == "passed":
            passed_task_ids_cot.append(results_cot[r]["task_id"])
    #print(passed_task_ids_cot)
    for r in results_scot:
        if results_scot[r]["result"] == "passed":
            passed_task_ids_scot.append(results_scot[r]["task_id"])
    #print(passed_task_ids_scot)
    for r in results_scot_few:
        if results_scot_few[r]["result"] == "passed":
            passed_task_ids_scot_few.append(results_scot_few[r]["task_id"])
    #print(passed_task_ids_scot_few)

    only_in_cot = set(passed_task_ids_cot) - set(passed_task_ids_scot) - set(passed_task_ids_scot_few)

    only_in_scot = set(passed_task_ids_scot) - set(passed_task_ids_cot) - set(passed_task_ids_scot_few)

    only_in_scot_few = set(passed_task_ids_scot_few) - set(passed_task_ids_cot) - set(passed_task_ids_scot)

    common_task_ids = set(passed_task_ids_cot) & set(passed_task_ids_scot) & set(passed_task_ids_scot_few)

    print("The task_id that exists in results_cot but not in results_scot and results_scot_few:", only_in_cot)
    print("The task_id that exists in results_scot but not in results_cot and results_scot_few:", only_in_scot)
    print("The task_id that exists in results_scot_few but not in results_cot and results_scot:", only_in_scot_few)
    print("The task_id that exists in all three results:", common_task_ids)
    """




