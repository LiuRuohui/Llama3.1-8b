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
    "216ca667-90ec-490b-978a-476ab39bcb79",
    "6c1efdc4-fcb7-45b8-962b-aec70649b27f",
    "1a6214e2-084a-4987-bb4e-48f770b8e068",
]
api_key_index = 0

# 初始化 OpenAI 客户端
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

# 分类类别列表
categories = [
    "Basic Algorithm Problems",
    "Mathematical Problems",
    "Data Structure Related Problems",
    "String Operations",
    "Logical Reasoning and Conditional Statements",
    "Complexity Analysis",
    "Specific Function Implementations",
    "Debugging and Code Fixing"
]

# 用于存储每个类别的任务 ID
categorized = {category: [] for category in categories}

def categorize_problems_with_id(text, task_id):
    """
    根据问题文本判断问题类别，并将任务 ID 分类
    """
    for category in categories:
        if category in text:
            categorized[category].append(task_id)

def code_generation(prompt, retries=100, delay=1):
    global api_key_index

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "give me the type of the problem based on your knowledge. Remember, you could only select one from the below list:\n Basic Algorithm Problems, Mathematical Problems, Data Structure Related Problems, String Operations, Logical Reasoning and Conditional Statements, Complexity Analysis, Specific Function Implementations, Debugging and Code Fixing.\n If you can't specify it, you can select two or three of them from the list."},
                    {"role": "user", "content": f"You only need to output the type of the question above the list without any other statements\n{prompt}"}
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

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        completion, token_count, prompt_given = code_generation(prompt)  # get response and token
        print(completion)
        elapsed_time = time.time() - start_time

        total_time += elapsed_time  # accumulate time
        total_tokens += token_count  # accumulate token

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })

            # 分类任务
            categorize_problems_with_id(completion, task_id)

            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    # 输出每个类别的任务 ID
    for category, task_ids in categorized.items():
        print(f"\n{category} task_ids: {task_ids}")

    # average time and token
    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("questionType.baseline.jsonl", generated_solutions)
