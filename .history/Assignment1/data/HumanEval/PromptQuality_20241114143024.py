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

# 定义不同的prompt模板
SIMPLE_PROMPT_TEMPLATE = "Generate code for the following task: {}"
DETAILED_PROMPT_TEMPLATE = "Please generate code to solve the following task. The task requires {} and should achieve {}. The environment is ipython. Task: {}"
EXAMPLE_PROMPT_TEMPLATE = "Here is an example of a similar task and its solution:\nExample Task: {}\nExample Solution: {}\nNow, generate code for the following task: {}"

# 初始化 OpenAI 客户端
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

def code_generation(prompt_template, task_description, additional_info=None, example_task=None, example_solution=None, retries=100, delay=5):
    global api_key_index

    for attempt in range(retries):
        try:
            if prompt_template == SIMPLE_PROMPT_TEMPLATE:
                prompt = prompt_template.format(task_description)
            elif prompt_template == DETAILED_PROMPT_TEMPLATE:
                prompt = prompt_template.format(additional_info, task_description)
            elif prompt_template == EXAMPLE_PROMPT_TEMPLATE:
                prompt = prompt_template.format(example_task, example_solution, task_description)

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

if __name__ == '__main__':
    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    # 用于存储不同prompt模板下的准确率等指标
    simple_prompt_results = {
        "total_time": 0,
        "total_tokens": 0,
        "correct_count": 0
    }
    detailed_prompt_results = {
        "total_time": 0,
        "total_tokens": 0,
        "correct_count": 0
    }
    example_prompt_results = {
        "total_time": 0,
        "total_tokens": 0,
        "correct_count": 0
    }

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        task_description = prompt
        additional_info = "specific functionality"  # 这里可根据具体问题补充更详细的要求信息
        example_task = "Another similar task description"  # 这里可根据具体情况补充示例任务描述
        example_solution = "Example code for the similar task"  # 这里可根据具体情况补充示例任务的解决方案

        # 使用简单prompt模板
        start_time = time.time()
        completion, token_count, prompt_given = code_generation(SIMPLE_PROMPT_TEMPLATE, task_description)
        elapsed_time = time.time() - start_time
        simple_prompt_results["total_time"] += elapsed_time
        simple_prompt_results["total_tokens"] += token_count

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            result = entry_point("zeroshot.baseline.jsonl", k="1", n_workers=4, timeout=3.0)
            if result["correct"]:
                simple_prompt_results["correct_count"] += 1
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

        # 使用详细prompt模板
        start_time = time.time()
        completion, token_count, prompt_given = code_generation(DETAILED_PROMPT_TEMPLATE, task_description, additional_info)
        elapsed_time = time.time() - start_time
        detailed_prompt_results["total_time"] += elapsed_time
        detailed_prompt_results["total_tokens"] += token_count

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            result = entry_point("zeroshot.baseline.jsonl", k="1", n_workers=4, timeout=3.0)
            if result["correct"]:
                detailed_prompt_results["correct_count"] += 1
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

        # 使用示例prompt模板
        start_time = time.time()
        completion, token_count, prompt_given = code_generation(EXAMPLE_PROMPT_TEMPLATE, task_description, example_task=example_task, example_solution=example_solution)
        elapsed_time = time.time() - start_time
        example_prompt_results["total_time"] += elapsed_time
        example_prompt_results["total_tokens"] += token_count

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            result = entry_point("zeroshot.baseline.jsonl", k="1", n_workers=4, timeout=3.0)
            if result["correct"]:
                example_prompt_results["correct_count"] += 1
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    # 计算不同prompt模板下的平均时间、平均token使用量和准确率
    simple_prompt_average_time = simple_prompt_results["total_time"] / len(problems)
    simple_prompt_average_tokens = simple_prompt_results["total_tokens"] / len(problems)
    simple_prompt_accuracy = simple_prompt_results["correct_count"] / len(problems)

    detailed_prompt_average_time = detailed_prompt_results["total_time"] / len(problems)
    detailed_prompt_average_tokens = detailed_prompt_results["total_tokens"] / len(problems)
    detailed_prompt_accuracy = detailed_prompt_results["correct_count"] / len(problems)

    example_prompt_average_time = example_prompt_results["total_time"] / len(problems)
    example_prompt_average_tokens = example_prompt_results["total_tokens"] / len(problems)
    example_prompt_accuracy = example_prompt_results["correct_count"] / len(problems)

    print(f"\nSimple Prompt:")
    print(f" Wall-Clock Time: {simple_prompt_average_time:.2f}s")
    print(f" Average Wall-Clock Time per Problem: {simple_prompt_average_time:.2f}s")
    print(f" Total Number of Generated Tokens: {simple_prompt_average_tokens:.2f}")
    print(f" Average Number of Generated Tokens per Problem: {simple_prompt_average_tokens:.2f}")
    print(f" Accuracy: {simple_prompt_accuracy:.2f}")

    print(f"\nDetailed Prompt:")
    print(f" Wall-Clock Time: {detailed_prompt_average_time:.2f}s")
    print(f" Average Wall-Clock Time per Problem: {detailed_prompt_average_time:.2f}s")
    print(f" Total Number of Generated Tokens: {detailed_prompt_average_tokens:.2f}")
    print(f" Average Number of Generated Tokens per Problem: {detailed_prompt_average_tokens:.2f}")
    print(f" Accuracy: {detailed_prompt_accuracy:.2f}")

    print(f"\nExample Prompt:")
    print(f" Wall-Clock Time: {example_prompt_average_time:.2f}s")
    print(f" Average Wall-Clock Time per Problem: {example_prompt_average_time:.2f}s")
    print(f" Total Number of Generated Tokens: {example_prompt_average_tokens:.2f}")
    print(f" Average Number of Generated Tokens per Problem: {example_prompt_average_tokens:.2f}")
    print(f" Accuracy: {example_prompt_accuracy:.2f}")

    write_jsonl("zeroshot.baseline.jsonl", generated_solutions)