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

def CoT_generation(prompt, cot_enabled=False, scot_enabled=False, retries=100, delay=1):
    global api_key_index

    if cot_enabled:
        # add CoT prompt
        cot_prompt = (
            "Let's solve this problem step by step. Begin by analyzing the problem requirements. "
            "Then break it down into smaller components. After that, implement a function that addresses the problem. "
        )
        prompt = f"{cot_prompt}\n\n{prompt}"

    if scot_enabled:
        # add SCoT prompt
        SCoT_prompt_fewshot= (
            "Here is a example for how to generate the SCoT using prompt given, you need to learn from it:\n"
            "```\nINPUT:\ndef first_Repeated_Char(str):\n"
            "    Write a python function to find the first repeated character in a given string.\n"
            "Please understand the requirement and write a rough solving process.\n It starts with a input - output structure.\n You should use three basic structures to build the solving process, including sequences, branches, and loops.\n"
            "The necessary details should be written in natural languages.\n\n"
            "OUTPUT:\n"
            "Input: str: a string\n"
            "Output: ch: a repeated character in str\n"
            "1: for each character ch in str:\n"
            "2: if ch appears more than once in str:\n"
            "3: return ch\n"
            "4: return None\n```\n"
        )
        SCoT_prompt = (
            "Please understand the requirement and write a rough solving process.\n"
            "It starts with a input - output structure.\n"
            "You should use three basic structures to build the solving process, including sequences, branches, and loops.\n"
            "The necessary details should be written in natural languages.\n\n"
        )
        #prompt = f"\n{prompt}\n Please understand the requirement and write a rough solving process.\n You should use three basic structures to build the solving process, including sequences, branches, and loops.\n The necessary details should be written in natural languages.\n Do not change the function name given.\n"
        #prompt = f"{SCoT_prompt}\n{prompt}\n."
        prompt = f"Please understand the requirement and write a rough solving process.\n You should use three basic structures to build the solving process, including sequences, branches, and loops.\n The necessary details should be written in natural languages.\n{prompt}\n."
        #print("Here is prompt\n", prompt)

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    #{"role": "system", "content": f"{SCoT_prompt_fewshot}\n"},
                    {"role": "user", "content": f"\n{SCoT_prompt_fewshot}\n Now solve this problem based on the prompt given:\n{prompt}\n"}
                ],
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

            return full_response, tokens, f"System:{SCoT_prompt}\n User:{prompt}"  # return full response and tokens and the prompt given

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0
def code_generation(CoT, prompt, retries=100, delay=1):
    global api_key_index
    SCoT_prompt_fewshot = (
        "Here is a example to generate the code from SCoT, you need to learn from it"
        "```\nINPUT:\ndef first_Repeated_Char(str):\n"
        " # Write a python function to find the first repeated character in a given string.\n"
        " Input: str: a string\n"
        " Output: ch: a repeated character in str\n"
        " 1: for each character ch in str:\n"
        " 2: if ch appears more than once in str:\n"
        " 3: return ch\n"
        " 4: return None\n"
        "# Please check the above solving process and write a code based on it. Note that the solving process may contain errors.\n"
        "\nOUTPUT:\n"
        "def first_Repeated_Char(str):"
        " h = {}\n"
        " for ch in str:\n"
        " if ch in h:\n"
        " return ch;\n"
        " else:\n"
        " h[ch] = 0\n"
        " return None```"
    )
    SCoT_prompt = (
         "# Please check the above solving process and write a code based on it.\n"
    )
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    #{"role": "system", "content": f"\n{SCoT_prompt}\n Take them as example!"},
                    #{"role": "user", "content": f"\n{SCoT_prompt}\n Generate the code directly without the example usage and Test function\n\n{CoT}"}
                    {"role": "user", "content": f"\n{SCoT_prompt_fewshot}\n Now solve this problem based on the prompt given:\n{CoT}\n Your task is check the above solving process and write a code based on it.\n Note that the solving process may contain errors\n"}
                ],
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

            return full_response, tokens, f"System: Environment:ipython \n Please check the above solving process and write a code based on it. Note that the solving process may contain errors.\n{CoT}"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0
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

if __name__ == '__main__':
    """
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
        completion, token_count1, prompt_given = CoT_generation(prompt, scot_enabled=True)
        total_tokens += token_count1
        #print("model output for SCoT\n",completion)
        completion, token_count2, prompt_given = code_generation(completion, prompt)
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
    write_jsonl("scot_baseline_fewshot.jsonl", generated_solutions)
    #result = entry_point("cot_baseline.jsonl", k="1", n_workers=4, timeout=5.0)
    #result = entry_point("scot_baseline_zeroshot.jsonl", k="1", n_workers=4, timeout=5.0)
    result = entry_point("scot_baseline_fewshot.jsonl", k="1", n_workers=4, timeout=5.0)
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

    # 找出在results_cot但不在results_scot和results_scot_few中的task_id
    only_in_cot = set(passed_task_ids_cot) - set(passed_task_ids_scot) - set(passed_task_ids_scot_few)

    # 找出在results_scot但不在results_cot和results_scot_few中的task_id
    only_in_scot = set(passed_task_ids_scot) - set(passed_task_ids_cot) - set(passed_task_ids_scot_few)

    # 找出在results_scot_few但不在results_cot和results_scot中的task_id
    only_in_scot_few = set(passed_task_ids_scot_few) - set(passed_task_ids_cot) - set(passed_task_ids_scot)

    # 找出三个列表中的共同task_id
    common_task_ids = set(passed_task_ids_cot) & set(passed_task_ids_scot) & set(passed_task_ids_scot_few)

    print("在results_cot但不在results_scot和results_scot_few中的task_id：", only_in_cot)
    print("在results_scot但不在results_cot和results_scot_few中的task_id：", only_in_scot)
    print("在results_scot_few但不在results_cot和results_scot中的task_id：", only_in_scot_few)
    print("在三个结果中都存在的task_id：", common_task_ids)

