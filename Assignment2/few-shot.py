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

# Initialize OpenAI client
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "input": "Write a function to add two numbers.",
        "output": (
            "def add_two_numbers(a, b):\n"
            "    Return the sum of two numbers.\n"
            "    return a + b\n"
        )
    },
    {
        "input": "Write a function to reverse a string.",
        "output": (
            "def reverse_string(s):\n"
            "    Return the reverse of the input string.\n"
            "    return s[::-1]\n"
        )
    },
]

def question_prompt(s):
    return f'Question: {s}'

def construct_few_shot_chats(n):
    """Construct few - shot chats based on the FEW_SHOT_EXAMPLES."""
    chats = []
    for example in FEW_SHOT_EXAMPLES[:n]:
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
def code_generation_fewshot(prompt, n=2, retries=100, delay=1):
    """Generate code using few - shot prompting with updated template."""
    global api_key_index

    few_shot_chats = construct_few_shot_chats(n)
    new_chat = [{"role": "user", "content": question_prompt(prompt)}]
    chats = few_shot_chats + new_chat

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="Meta-Llama-3.1-8B-Instruct",
                messages=chats,
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

            return full_response, tokens, chats  # return full response, tokens and chats

        except openai.RateLimitError:
            print(f"Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)
            client.api_key = API_KEYS[api_key_index]
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

if __name__ == '__main__':
    problems = read_problems()
    prompts = []
    task_ids = []
    generated_solutions = []
    total_time = 0
    total_tokens = 0

    for p in problems:
        task_id = problems[p]["task_id"]
        prompt = problems[p]["prompt"]
        prompts.append(prompt)
        task_ids.append(task_id)

    for i in range(len(task_ids)):
        task_id = task_ids[i]
        prompt = prompts[i]

        start_time = time.time()
        completion, token_count, chats = code_generation_fewshot(prompt)  # use few - shot prompting
        #print("\n\n", clean_the_wrap(completion))
        elapsed_time = time.time() - start_time

        total_time += elapsed_time
        total_tokens += token_count

        if completion:
            generated_solutions.append({
                "task_id": task_id,
                "input": prompt,
                "prompt": chats,
                "output": clean_the_wrap(completion),
                "elapsed_time": elapsed_time,
                "token_count": token_count
            })
            print(f"Task ID: {task_id}, Time: {elapsed_time:.2f}s, Tokens: {token_count}")

    average_time = total_time / len(prompts) if prompts else 0
    average_tokens = total_tokens / len(prompts) if prompts else 0

    print(f"\n Wall - Clock Time: {total_time:.2f}s")
    print(f"\n Average Wall - Clock Time per Problem: {average_time:.2f}s")
    print(f"\n Total Number of Generated Tokens: {total_tokens:.2f}")
    print(f"\n Average Number of Generated Tokens per Problem: {average_tokens:.2f}")

    write_jsonl("fewshot.baseline.jsonl", generated_solutions)
    result = entry_point("fewshot.baseline.jsonl", k="1", n_workers=4, timeout=5.0)