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
]
api_key_index = 0

# 初始化 OpenAI 客户端
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])


def code_generation(prompt, retries=100, delay=5):
    global api_key_index

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Environment: ipython"},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                stream_options={"include_usage": True}
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

            return full_response, tokens, f"System:Environment: ipython\n User:{prompt}"  # return full response and tokens and the prompt given

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)  # change the API
            client.api_key = API_KEYS[api_key_index]  # update the API
            time.sleep(delay)  # wait
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0


def generate_with_prompt_quality_1():
    """
    the function without any prompt only with the prompt from the HumanEval
    """
    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = f"{problems[p]['prompt']}"
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        completion, token_count, prompt_given = code_generation(prompt)  # get response and token
        elapsed_time = time.time() - start_time

        total_time += elapsed_time  # accumulate time
        total_tokens += token_count  # accumulate token

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    # average time and token
    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("prompt_quality_1.baseline.jsonl", generated_solutions)
    result_quality_1 = entry_point("prompt_quality_1.baseline.jsonl", k="1", n_workers=4, timeout=5.0)
    return result_quality_1


def generate_with_prompt_quality_2():
    """
    the normal quality of the prompt
    """
    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = f"Analyze the task requirements unambiguously, output python formats directly\n{problems[p]['prompt']}"
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        completion, token_count, prompt_given = code_generation(prompt)  # get response and token
        elapsed_time = time.time() - start_time

        total_time += elapsed_time  # accumulate time
        total_tokens += token_count  # accumulate token

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    # average time and token
    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("prompt_quality_2.baseline.jsonl", generated_solutions)
    result_quality_2 = entry_point("prompt_quality_2.baseline.jsonl", k="1", n_workers=4, timeout=5.0)
    return result_quality_2


def generate_with_prompt_quality_3():
    """
    the best quality of the code
    """
    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = f"You are a good assistant in solving the code generation problems, please solve it use the prompts provided below\n{problems[p]['prompt']}"
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        completion, token_count, prompt_given = code_generation(prompt)  # get response and token
        elapsed_time = time.time() - start_time

        total_time += elapsed_time  # accumulate time
        total_tokens += token_count  # accumulate token

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    # average time and token
    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("prompt_quality_3.baseline.jsonl", generated_solutions)
    result_quality_3 = entry_point("prompt_quality_3.baseline.jsonl", k="1", n_workers=4, timeout=5.0)
    return result_quality_3


if __name__ == '__main__':
    result_1 = generate_with_prompt_quality_1()
    result_2 = generate_with_prompt_quality_2()
    result_3 = generate_with_prompt_quality_3()