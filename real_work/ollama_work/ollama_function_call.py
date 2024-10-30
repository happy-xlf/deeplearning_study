#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
@File    :   ollama_function_call.py
@Time    :   2024/10/14 22:15:16
@Author  :   Lifeng Xu 
@desc :   
'''
import openai
import os
from math import *
import json
from math import *
import requests
import logging
# 设置日志记录配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

def get_completion(messages, model="qwen2.5:0.5b"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,# 模型输出的随机性，0 表示随机性最小
        tools=[{
            "type": "function",
            "function": {
                "name": "add_contact",
                "description": "添加联系人",
                "parameters": {
                    "type": "object",
                    "properties": {
                            "name": {
                            "type": "string",
                            "description": "联系人姓名"
                            },
                    "address": {
                            "type": "string",
                            "description": "联系人地址"
                            },
                    "tel": {
                            "type": "string",
                            "description": "联系人电话"
                            },
                    }
                }
             }
        }],
    )
    return response.choices[0].message
 
 
 
def function_test_promopt():
    prompt = "请寄给上海黄浦区科学会堂的老范，地址是上海市黄浦区科学会堂国际厅3楼，电话18888888888。"
    messages = [
        {"role": "system", "content": "你是一个联系人录入员。"},
        {"role": "user", "content": prompt}
        ]
    response = get_completion(messages)
    logging.info("====GPT回复====")
    logging.info(response)
    args = json.loads(response.tool_calls[0].function.arguments)
    logging.info("====函数参数====")
    logging.info(args)
 
if __name__ == '__main__':
 
    function_test_promopt()