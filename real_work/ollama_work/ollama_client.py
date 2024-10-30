#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   ollama_client.py
@Time    :   2024/10/14 18:04:36
@Author  :   Lifeng Xu 
@desc :   
'''

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '你是谁？',
        }
    ],
    model='qwen2.5:7b',
).choices[0].message.content


# completion = client.completions.create(
#     model="llama3.2",
#     prompt="Say this is a test",
# )

# list_completion = client.models.list()
# print(list_completion)
print(chat_completion)
# model = client.models.retrieve("llama3.2")

# embeddings = client.embeddings.create(
#     model="all-minilm",
#     input=["why is the sky blue?", "why is the grass green?"],
# )
