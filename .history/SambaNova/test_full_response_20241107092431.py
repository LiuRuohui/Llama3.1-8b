from openai import OpenAI
import time

start_time = time.time()
client = OpenAI(base_url="https://api.sambanova.ai/v1", api_key="31a55a95-3f5c-483b-9a35-5fa473a6006a")

completion = client.chat.completions.create(
    model="Meta-Llama-3.1-405B-Instruct",
    messages=[
        {"role": "system", "content": "You are good programmer"},
        {"role": "user", "content": "tell me about how to do word embedding"}
    ],
    stream=True,
    stream_options={"include_usage": True}
)
full_response = ""
total_tokens = 0

# 逐个处理返回的 chunk
for chunk in completion:
    # 检查是否存在 choices
    if chunk.choices:
        delta = chunk.choices[0].delta
        if hasattr(delta, 'content'):
            full_response += delta.content

# 在最后一行处理使用情况
if chunk.usage:  # 确保存在 usage 信息
    print(chunk.usage)
    total_tokens += chunk.usage.completion_tokens

end_time = time.time()-start_time

# 输出结果
print(f"生成的文本: {full_response}")
print(f"生成的令牌数: {total_tokens}")
print(f"时间: {end_time}")
# print(completion.choices[0].message.content, completion.usage.completion_tokens)



# If you are using the original script in the .pdf, you can set streaming to false, then print(completion.choices[0].message.content)