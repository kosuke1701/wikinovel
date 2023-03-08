"""
Copyright (c) 2023 Kosuke Akimoto
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from argparse import ArgumentParser
import json
from time import sleep

from langchain.llms import OpenAI
from tqdm import tqdm

from wikinovel.wiki import *

class ChunkProcessor:
    def __init__(self, llm=None):
        self.entity_extraction_chain = load_entity_extraction_chain(llm=llm)
        self.entity_new_info_chain = load_entity_new_info_chain(llm=llm, prompt_fn=DEFAULT_ROOT_DIR+"prompts/entity_new_info_prompt_chat.json")
    
    def process_chunk(self, text):
        ents = self.entity_extraction_chain.run(
            chunk=text
        )
        ent2info = {}
        for ent in ents:
            info = self.entity_new_info_chain.run(
                chunk=text,
                entity=ent
            )
            if info.strip() == "N/A":
                continue
            ent2info[ent] = info

        return ent2info

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--novel_text_fn")
    parser.add_argument("--save_fn")
    parser.add_argument("--chunk_length_threshold", type=int, default=1000)

    args = parser.parse_args()

    llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024)
    processor = ChunkProcessor(llm=llm)

    with open(args.novel_text_fn) as h:
        novel = h.read()

    text = ""
    with open(args.save_fn, "w") as h:
        def do_process():
            ent2info = processor.process_chunk(text)
            print(ent2info)

            output = {
                "chunk": text,
                "info": ent2info
            }
            h.write(f"{json.dumps(output, ensure_ascii=False)}\n")
            h.flush()
            sleep(1)

        for line in tqdm(novel.split("\n")):
            if len(text) + len(line) > args.chunk_length_threshold:
                do_process()
                text = ""
            text += line
        else:
            if len(text) > 0:
                do_process()
            
