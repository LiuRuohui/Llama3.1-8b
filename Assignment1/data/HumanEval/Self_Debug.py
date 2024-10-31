import time
from openai import OpenAI
import openai
from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from evaluate_functional_correctness import entry_point
from test_single import test_generated_code_with_result
from extract_one_test_case import extract_one_test_case

# API Keys List
API_KEYS = [
    "bab5a926-5245-4843-a03d-d98b57a0c644",
    "31a55a95-3f5c-483b-9a35-5fa473a6006a",
]
api_key_index = 0
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

def switch_api_key():
    global api_key_index, client
    api_key_index = (api_key_index + 1) % len(API_KEYS)
    client.api_key = API_KEYS[api_key_index]
    print(f"Switched to new API key: {API_KEYS[api_key_index]}")

def generate_initial_code(prompt, test, retries=100, delay=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Environment:ipython \n You are an expert programming assistant."},
                    {"role": "user", "content": f"{prompt}+{test}"}
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

            return full_response, tokens, f"System: Environment:ipython \n You are an expert programming assistant\n User:{prompt+test}\n"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def regenerate_code(code, retries=100, delay=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Environment:ipython"},
                    {"role": "user", "content": f"{code}\n Is the code above correct? If not, fix the code directly without anything more"}
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

            return full_response, tokens, f"System:Environment:ipython\n {code}\n Is the code above correct? If not, fix the code directly without anything more"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def self_debug_generation(code, test, message, retries=100, delay=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": f"{code}\n The code above fails the given test:\n {test}+\n+{message}\n Please fix the code directly without any comment especially ``` python."}
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

            return full_response, tokens, f"{code}\n The code above fails the given test:\n {test}+\n+{message}\n Please fix the code directly without any comment especially ``` python."

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def self_debug(task_id, prompt, test, entry_point, max_iterations=5):
    """Self-Debug"""
    # step 1
    code, token0, help = generate_initial_code(prompt, test)
    # step 2&3
    total_tokens = token0
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")
        status, message, detailed_message= test_generated_code_with_result(task_id, prompt, code, entry_point, test, timeout=5.0)

        if status == "passed":
            print(f"{task_id} Code executed successfully!")
            code1, token1, help1=regenerate_code(code)
            print("check another time")
            status1, message1, detailed_message1= test_generated_code_with_result(task_id, prompt, code1,  entry_point, test, timeout=5.0)
            if status1 == "passed":
                return code1, token0+token1, help1
            else:
                return code, token0+token1, help
        else:
            print(f"{task_id}Execution failed with error: {message}")
            code, token2, help2 = self_debug_generation(code, test, message)
            total_tokens += token2
            help = help2
    return code, total_tokens, help2

if __name__ == "__main__":
    problems = read_problems()
    generated_solutions = []

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        test = problems[p]["test"]
        entrypoint = problems[p]["entry_point"]
        start_time = time.time()
        completion, total_token, prompt_given = self_debug(task_id, prompt, extract_one_test_case(test), entrypoint)
        elapsed_time = time.time() - start_time
        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": prompt_given,
                "output": completion,
                "elapsed_time": elapsed_time,
                "token_count": total_token
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {total_token}")
    write_jsonl("Self_Debug.jsonl", generated_solutions)
    final_result = entry_point("Self_Debug.jsonl", k="1", n_workers=4, timeout=5.0)

