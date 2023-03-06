"""
Copyright (c) 2023 Kosuke Akimoto
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

import os
import re

from langchain.chains import LLMChain, TransformChain, SimpleSequentialChain, SequentialChain
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

def load_select_wiki_section_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/select_wiki_section_prompt.json"
):
    """
    entity, index, info -> sections
    """
    if llm is None:
        llm = OpenAI(max_tokens=1024)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm)

    def func_parse(inputs):
        text = inputs["text"]
        parts = {}
        sections = {}
        for line in text.split("\n"):
            m = re.search(r"Section (?P<idx>[0-9]+): (?P<names>.+)", line)
            if m is not None:
                idx = m.group("idx")
                names = m.group("names")
                names = names.split(" / ")
                sects = []
                for name in names:
                    m = re.search(r"(?P<sec_idx>[0-9.]+) (?P<name>.+)", name)
                    if m is not None:
                        sec_idx = m.group("sec_idx")
                        name = m.group("name")
                        sec_idx = [int(num) for num in sec_idx.split(".") if len(num) > 0]
                        sects.append((sec_idx, name))
                sections[idx] = sects
            
                continue

            m = re.search(r"Part (?P<idx>[0-9]+): (?P<content>.+)", line)
            if m is not None:
                idx = m.group("idx")
                content = m.group("content")

                parts[idx] = content
        
        outputs = []
        for idx in parts.keys() & sections.keys():
            outputs.append((sections[idx], parts[idx]))
        
        return {"sections": outputs}

    parse_chain = TransformChain(transform=func_parse, input_variables=["text"], output_variables=["sections"])

    chain = SequentialChain(
        chains=[chain, parse_chain],
        input_variables=chain.input_keys,
        output_variables=["sections"]
    )

    return chain

def load_update_wiki_section_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/update_wiki_section_prompt.json"
):
    """
    section, info -> text
    """
    if llm is None:
        llm = OpenAI(max_tokens=1024)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm)

    return chain

def load_entity_extraction_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/entity_extraction_prompt.json"
):
    """
    chunk -> entities
    """
    if llm is None:
        llm = OpenAI(max_tokens=1024)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm)

    def func_parse(inputs):
        text = inputs["text"]
        return {"entities": [ent.strip() for ent in text.split(",")]}
    parse_chain = TransformChain(transform=func_parse, input_variables=["text"], output_variables=["entities"])

    chain = SequentialChain(
        chains=[chain, parse_chain],
        input_variables=chain.input_keys,
        output_variables=parse_chain.output_keys
    )

    return chain

def load_split_wiki_section_chain(
    llm=None,
    prompt_fn=DEFAULT_ROOT_DIR + "prompts/split_wiki_section_prompt.json",
    verbose=False,
):
    """
    entity, index, info -> sections
    """
    if llm is None:
        llm = OpenAI(max_tokens=1024)
    prompt = load_prompt(prompt_fn)
    chain = LLMChain(prompt=prompt, llm=llm, verbose=verbose)

    def func_parse(inputs):
        text = inputs["text"]
        if verbose:
            print("Intermediate text in split_wiki_section_chain:")
            print(text)
        parts = {}
        titles = {}
        for line in text.split("\n"):
            m = re.search(r"Title (?P<idx>[0-9]+): (?P<names>.+)", line)
            if m is not None:
                idx = m.group("idx")
                names = m.group("names")
                titles[idx] = names
            
                continue

            m = re.search(r"Part (?P<idx>[0-9]+): (?P<content>.+)", line)
            if m is not None:
                idx = m.group("idx")
                content = m.group("content")

                parts[idx] = content
        
        outputs = []
        for idx in parts.keys() & titles.keys():
            outputs.append((titles[idx], parts[idx]))
        
        return {"sections": outputs}

    parse_chain = TransformChain(transform=func_parse, input_variables=["text"], output_variables=["sections"])

    chain = SequentialChain(
        chains=[chain, parse_chain],
        input_variables=chain.input_keys,
        output_variables=["sections"]
    )

    return chain