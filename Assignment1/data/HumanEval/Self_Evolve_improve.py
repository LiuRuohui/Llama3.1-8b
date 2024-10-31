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

def generate_knowledge(prompt,retries=100, delay=5):
    # Generate the necessary knowledge from the input prompts
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": f"```\n{prompt}\n```\n For the above question, could you briefly teach me how to solve it step by step in natural language?\nDon't write the code in this step"}
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
            return full_response, tokens, f"```\n{prompt}\n```\n For the above question, could you briefly teach me how to solve it step by step in natural language?\nDon't write the code in this step"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def generate_initial_code(knowledge, prompt, retries=100, delay=5):
    # Generate an initial code solution based on the generated knowledge and problem description
    prompt_edited = "Based on the above knowledge, help me complete the prompt.\n Be attention, you should only output the codes without any explanation and natural language.\n Wrap your code with ``` "
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Environment:ipython"},
                    {"role": "user", "content": f"{knowledge}\n+{prompt}+\n{prompt_edited}"}
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

            return full_response, tokens, f"{knowledge}\n+{prompt}+\n{prompt_edited}"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def self_refine(code, error, test, retries=100, delay=5):
    edited_prompt = f"```{code}```\n When I run the above code, the result of \n{test}\n is \n{error}\n Help me refine the code.\n You should only output the codes without any explanation and natural language.\n Wrap your code with ```"
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": edited_prompt}
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

            return full_response, tokens, edited_prompt

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def self_refine_syntax(code, error, test, retries=100, delay=5):
    edited_prompt = f"```{code}```\n+ When I run {test}, I meet syntax error which represent as:\n {error}\n Help me refine the code.\n you should only output the codes without any explanation and natural language.\n Wrap your code with ``` "
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": edited_prompt}
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

            return full_response, tokens, edited_prompt

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def regenerate_code(prompt, code, error, detailed_error, retries=100, delay=5):
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": "Environment:ipython \n You are an expert programming assistant."},
                    {"role": "user", "content": f"{prompt}\n You generated the code \n{code}\n with error message \n{error}\n{detailed_error}\nRegenerate the code please. \n You should only output the codes without any explanation and natural language.\n Wrap your code with ```"}
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

            return full_response, tokens, f"{prompt}\n You generated the code \n{code}\n with error message \n{error}\n{detailed_error}\nRegenerate the code please. \n You should only output the codes without any explanation and natural language.\n Wrap your code with ```"

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, retrying after {delay} seconds...")
            switch_api_key()
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0
def clean_the_wrap(code):
    # Remove ``` markers and the word 'python'
    cleaned_code = code.strip("```").replace("```", "").replace("python", "").strip()
    return cleaned_code

def self_evolve_improve(task_id, prompt, test, entry_point, max_iterations=5):
    """Self-Evolve-Improve"""
    knowledge, token0, help = generate_knowledge(prompt)
    #print(f"This is {task_id} knowledge\n", knowledge)
    code, token1, help1 = generate_initial_code(knowledge, prompt)
    #print(f"This is {task_id} code\n", code)
    total_tokens = token0
    final_help = help1
    error_temp = ""
    detailed_error_temp = ""
    # Self-Refine
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")
        random_test = extract_one_test_case(test)
        #print("random test generated\n")
        status, error, detailed_error = test_generated_code_with_result(
            task_id, prompt, clean_the_wrap(code), entry_point, random_test, timeout=5.0
        )

        if status == "passed":
            print(f"{task_id} Code executed successfully!")
            return clean_the_wrap(code), token0 + token1, final_help
        else:
            print(f"{task_id} Execution failed with error: {error}")
            if "syntax" in error:
                code, token2, help2 = self_refine_syntax(code, error, random_test)
                total_tokens += token2
                final_help = help2
                error_temp = error
                detailed_error_temp = detailed_error
            else:
                if error_temp == error and detailed_error_temp == detailed_error:
                    print(f"{task_id} Error is unchanged, regenerate the code.")
                    code, token3, help3 = regenerate_code(prompt, code, error, detailed_error)
                    total_tokens += token3
                    final_help = help3
                else:
                    code, token4, help4 = self_refine(code, error, random_test)
                    total_tokens += token4
                    final_help = help4
                    error_temp = error
                    detailed_error_temp = detailed_error

    return clean_the_wrap(code), total_tokens, final_help


if __name__ == "__main__":
    problems = read_problems()
    generated_solutions = []
    generated_knowledges = []
    task_ids = []
    total_time = 0  # Initial Total Time
    total_tokens = 0  # Initial Total Token

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        test = problems[p]["test"]
        entrypoint = problems[p]["entry_point"]
        task_ids.append(task_id)
        start_time = time.time()
        completion, total_token, prompt_given = self_evolve_improve(task_id, prompt, test, entrypoint)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time  # accumulate time
        total_tokens += total_token  # accumulate token

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

    average_time = total_time / len(task_ids) if task_ids else 0
    average_tokens = total_tokens / len(task_ids) if task_ids else 0

    print(f"\n Wall-Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall-Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("Self_Evolve_improve.jsonl", generated_solutions)
    final_result = entry_point("Self_Evolve_improve.jsonl", k="1", n_workers=4, timeout=5.0)
