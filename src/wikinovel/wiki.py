"""
Copyright (c) 2023 Kosuke Akimoto
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

import os
import re

from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import load_prompt

DEFAULT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def load_entity_new_info_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/entity_new_info_prompt.json",
    output_key="text"
):
    """
    与えられた文章中の指定されたエンティティに関する情報を抜き出す。Wikipediaに載せるような感じで。

    chunk, entity -> text (default)
    """
    if llm is None:
        llm = OpenAI(max_tokens=512)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm, output_key=output_key)
    return chain

def load_entity_alias_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/entity_alias_prompt.json"
):
    """
    与えられたエンティティに対するエイリアスのリストを生成。
    ※数回実行してunionを取った方がいいかも。

    entity -> alias_names
    """
    if llm is None:
        llm = OpenAI(max_tokens=512)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm)

    def parse_func(inputs):
        text = inputs["text"]
        outputs = []
        for line in text.split("\n"):
            m = re.search("Alias name [0-9]+: (?P<content>.+)", line)
            if m is not None:
                outputs.append(m.group("content"))
        return {"alias_names": outputs}
    
    parse_chain = TransformChain(input_variables=["text"], output_variables=["alias_names"], transform=parse_func)
    chain = SimpleSequentialChain(chains=[chain, parse_chain], input_key="entity")

    return chain
