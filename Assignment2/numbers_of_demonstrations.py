import time
from baseline import HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
from evaluate_functional_correctness import entry_point
from openai import OpenAI
import openai

# API Keys List
API_KEYS = [
    "eed9ffb7-df05-46b4-b7c1-cf65ce498eec",
    "e6835036-7961-4149-ba11-0430875b8a3b",
    "227ba7ed-3adb-4655-9485-6cfb0eb91429",
    "bab5a926-5245-4843-a03d-d98b57a0c644",
    "31a55a95-3f5c-483b-9a35-5fa473a6006a",
]
api_key_index = 0

# Initialize OpenAI client
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key=API_KEYS[api_key_index])

# Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n"
            "    given threshold.\n\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            "    \"\"\"\n"
        ),
        "output": (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if abs(numbers[i] - numbers[j]) < threshold:\n"
            "                return True\n"
            "    return False\n"
        )
    },
    {
        "input": (
            "from typing import List\n\n"
            "def find_largest_element(numbers: List[int]) -> int:\n"
            "    \"\"\" Return the largest element in the given list of numbers.\n\n"
            "    >>> find_largest_element([1, 2, 3, 4, 5])\n"
            "    5\n"
            "    >>> find_largest_element([-1, -5, -3])\n"
            "    -1\n"
            "    \"\"\"\n"
        ),
        "output": (
            "def find_largest_element(numbers: List[int]) -> int:\n"
            "    return max(numbers)\n"
        )
    },
    {
        "input": (
            "from typing import List\n\n"
            "def get_even_numbers(numbers: List[int]) -> List[int]:\n"
            "    \"\"\" Return a list of all even numbers from the given list.\n\n"
            "    >>> get_even_numbers([1, 2, 3, 4, 5])\n"
            "    [2, 4]\n"
            "    >>> get_even_numbers([10, 15, 20, 25])\n"
            "    [10, 20]\n"
            "    \"\"\"\n"
        ),
        "output": (
            "def get_even_numbers(numbers: List[int]) -> List[int]:\n"
            "    return [num for num in numbers if num % 2 == 0]\n"
        )
    }
]


def clean_the_wrap(code):
    # Remove ``` markers and the word 'python'
    start_index = code.find('```')
    if start_index != -1:
        end_index = code.find('```', start_index + 3)
        if end_index != -1:
            code = code[start_index + 3:end_index].replace("python", "")
    return code.strip()

def question_prompt(s):
    return f'Question: {s}'

def construct_few_shot_chats(n):
    """Construct few-shot chats based on the FEW_SHOT_EXAMPLES."""
    chats = []
    for example in FEW_SHOT_EXAMPLES[:n]:
        chats.append({"role": "user", "content": question_prompt(example["input"])})
        chats.append({"role": "assistant", "content": example["output"]})
    return chats

def code_generation_fewshot(prompt, n=3, retries=100, delay=1):
    """Generate code using few-shot prompting with updated template."""
    global api_key_index

    few_shot_chats = construct_few_shot_chats(n)
    new_chat = [{"role": "user", "content": question_prompt(prompt)}]
    chats = few_shot_chats + new_chat
    #print("CHATS\n",chats)

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

            return full_response, tokens, chats  # Return full response, tokens, and chats

        except openai.RateLimitError:
            print("Request rate limit exceeded, switching API key...")
            api_key_index = (api_key_index + 1) % len(API_KEYS)
            client.api_key = API_KEYS[api_key_index]
            time.sleep(delay)
        except Exception as e:
            print(f"Error occurred while generating code, retrying after {delay} seconds...")
            time.sleep(delay)

    return None, 0

def experiment_with_demos(problems, max_demos=3):
    """Run experiments with varying numbers of demonstrations."""
    results = []
    for demo_count in range(1, max_demos + 1):  # Test with 0 to max_demos examples
        total_time = 0
        total_tokens = 0
        generated_solutions = []

        for p in problems:
            task_id = problems[p]["task_id"]
            prompt = problems[p]["prompt"]
            print(task_id)
            start_time = time.time()
            completion, token_count, chats = code_generation_fewshot(prompt, n=demo_count)
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

        avg_time = total_time / len(problems) if problems else 0
        avg_tokens = total_tokens / len(problems) if problems else 0

        # Evaluate accuracy
        output_file = f"fewshot_{demo_count}.jsonl"
        write_jsonl(output_file, generated_solutions)
        accuracy = entry_point(output_file, k="1", n_workers=4, timeout=5.0)

        results.append({
            "demo_count": demo_count,
            "total_time": total_time,
            "avg_time": avg_time,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens
        })
        print(f"Demo Count: {demo_count}, Total Time: {total_time}s, Average Time: {avg_time:.2f}s, Total Tokens: {total_tokens:.2f}, Avg Tokens: {avg_tokens:.2f},")

    return results

if __name__ == '__main__':
    # Load problems
    problems = read_problems()

    # Run experiment with varying numbers of demonstrations
    max_demos = len(FEW_SHOT_EXAMPLES)
    print("MAXDEMOS", max_demos)
    experiment_results = experiment_with_demos(problems, max_demos=max_demos)

    print("\nExperiment Results:")
    for result in experiment_results:
        print(result)
