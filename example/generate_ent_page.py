"""
Copyright (c) 2023 Kosuke Akimoto
Released under the MIT license
https://opensource.org/licenses/mit-license.php
"""

from argparse import ArgumentParser
from collections import defaultdict
import json
from time import sleep

from langchain.llms import OpenAI
from tqdm import tqdm

from wikinovel.wiki import *

class Section:
    def __init__(self, name=""):
        self.childs = []
        self.text = ""
        self.name = name
        self.parent = None
    
    def get_index(self, depth=None):
        sections = []
        if depth is not None:
            depth -= 1
        for i_child, child in enumerate(self.childs):
            sections.append(((i_child + 1,), child.name, child.text, child))
            if depth is None or depth > 0:
                for idxs, name, text, sect in child.get_index(depth=depth):
                    sections.append(((i_child + 1, *idxs), name, text, sect))
        return sections

    def get_ancestors(self, depth=None):
        ancestors = [[[], self.name, self.text, self]]
        if depth is not None:
            depth -= 1
        if self.parent is not None:
            if depth is None or depth > 0:
                ancestors += self.parent.get_ancestors(depth=depth)
            self_index = self.parent.childs.index(self)
            ancestors[0][0] = [*ancestors[1][0], self_index + 1]

        return ancestors
    
    def to_plain_text(self):
        sections = self.get_index()
        sections = [f"{'#'*len(idxs)} {'.'.join(str(idx) for idx in idxs)}. {name}\n{text}" for idxs, name, text, _ in sections]
        sections.insert(0, self.text)
        return "\n\n".join(sections)

    def __len__(self):
        return len(self.childs)

    def __getitem__(self, idx):
        return self.childs[idx]
    
    def add_blank_section(self, name=""):
        sect = Section(name=name)
        sect.parent = self
        self.childs.append(sect)
        return sect
    
class Wiki:
    def __init__(self, llm=None):
        self.pages = []
        self.aliases = []
        self.name2pages = {}

        self.select_wiki_section_chain = load_select_wiki_section_chain(llm=llm, prompt_fn=DEFAULT_ROOT_DIR+"prompts/select_wiki_section_local_prompt.json")
        self.update_wiki_section_chain = load_update_wiki_section_chain(llm=llm)
        self.split_wiki_section_chain = load_split_wiki_section_chain(llm=llm)
    
    def _split_page(self, ent_name, section_position_text, page):
        info = page.text
        page.text = ""

        output = self.split_wiki_section_chain.run(
            entity=ent_name,
            index=section_position_text,
            info=info
        )

        for title, subinfo in output:
            subsec = page.add_blank_section(title)
            subsec.text = subinfo
    
    def split_page(self, ent_name, page: Section):
        ancestors = page.get_ancestors()

        texts = []
        for idxs, title, _, _ in ancestors[:-1][::-1]: # last one is root (page)
            texts.append(f"{'.'.join(str(idx) for idx in idxs)}. {title}")
        position_text = "\n".join(texts)

        self._split_page(ent_name, position_text, page)
    
    def update_page(self, ent_name, page, new_info):
        if len(page) > 0:
            self._update_page_split(ent_name, page, new_info)
        else:
            self._update_page_merge(ent_name, page, new_info)
    
    def _update_page_merge(self, ent_name, page: Section, new_info):
        prev_text = page.text
        new_text = self.update_wiki_section_chain.run(
            section=prev_text,
            info=new_info
        )
        page.text = new_text
    
    def _update_page_split(self, ent_name, page: Section, new_info):
        index = page.get_index(depth=1)
        index_text = "\n".join(".".join(str(i) for i in idxs)+"."+" "+name for idxs, name, _, _ in index)
        existing_idxs = {idxs[0]:name for idxs, name, _, _ in index}

        sections = self.select_wiki_section_chain.run(
            entity=ent_name,
            index=index_text,
            info=new_info
        )

        # print(index_text)
        # print(sections)

        target_positions = set()
        idx2name = {}
        for positions, _ in sections:
            for sec_idxs, name in positions:
                if len(sec_idxs) != 1:
                    print("Illegal output format. Ignore. {sec_idxs} {name}")
                    continue
                target_positions.add(sec_idxs[0])
                idx2name[sec_idxs[0]] = name

        # print(target_positions)
        # print(idx2name)
        # print(existing_idxs)
        old2new = {}
        for idx in sorted(target_positions):
            target_idx = idx
            while target_idx > 1 and target_idx-1 not in (idx2name.keys() | existing_idxs.keys()):
                target_idx -= 1
            
            idx2name[target_idx] = idx2name.pop(idx)
            old2new[idx] = target_idx
        target_positions = idx2name.keys()

        for idx in sorted(target_positions):
            while len(page) < idx:
                page.add_blank_section(name=idx2name.get(idx, ""))
        
        # print(target_positions)
        # print(idx2name)
        
        idx2newinfo = defaultdict(str)
        for positions, sub_info in sections:
            target_positions = set()
            for sec_idxs, name in positions:
                if len(sec_idxs) != 1:
                    continue
                target_positions.add(sec_idxs[0])
            
            for idx in target_positions:
                if idx in old2new:
                    idx = old2new[idx]
                idx2newinfo[idx] += " " + sub_info
        
        # print(idx2newinfo)
        
        for idx, sub_info in idx2newinfo.items():
            self.update_page(ent_name, page[idx-1], sub_info)

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--ent_info_fn")
    parser.add_argument("--target_ent")
    parser.add_argument("--output_fn")

    parser.add_argument("--enable_subsection", action="store_true")
    parser.add_argument("--section_length_threshold", type=int, default=150)
    
    args = parser.parse_args()

    page = Section()
    wiki = Wiki(llm=OpenAI(model_name="gpt-3.5-turbo", max_tokens=1024))

    data = []
    with open(args.ent_info_fn) as h:
        for line in h.read().split("\n"):
            if len(line) == 0:
                continue
            data.append(json.loads(line))
    
    for chunk in tqdm(data):
        ent2info = chunk["info"]
        for ent, info in ent2info.items():
            if args.target_ent == ent:
                print("New info:")
                print(info)
                wiki.update_page(args.target_ent, page, info)
                all_pages = [page, *[subsect for _,_,_,subsect in page.get_index()]]
                for subsect in all_pages:
                    if len(subsect.text.split(" ")) > args.section_length_threshold:
                        wiki.split_page(args.target_ent, subsect)
                print(page.to_plain_text())
    
    with open(args.output_fn, "w") as h:
        h.write(page.to_plain_text())