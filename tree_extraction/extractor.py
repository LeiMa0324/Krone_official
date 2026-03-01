from __future__ import annotations


import asyncio
import os.path
from typing import Any
from tree_extraction.EXTRACT_PROMPTS import *
import logging
from llm.llm import LLM, LLMOutput
import pandas as pd
pd.set_option('display.max_columns', 5)
import re
import json
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


# DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_ENTITY_TYPES = ["SYSTEM MODULE", "OBJECT"]
DEFAULT_STATUS_TYPES = ["MODULE STATUS", "PROCESS STATUS", "OBJECT STATUS"]
DEFAULT_ACTION_TYPES = ["MODULE ACTION" "PROCESS ACTION", "OBJECT ACTION"]

# modules: packetResponder, host,
# process: connection, verification,
# object: block,
def to_camel_case(sentence):
    if isinstance(sentence, float):
        return 'none'
    if sentence in ['none', 'None']:
        return 'none'
    # Split the sentence into words using spaces
    words = sentence.split()
    # Capitalize the first letter of each word and lowercase the rest
    camel_case_sentence = ''.join(word.capitalize() for word in words)
    return camel_case_sentence

class Extractor(object):
    def __init__(self, llm,
                 entity_types: str | None = None,
                 tuple_delimiter_key: str | None = None,
                 record_delimiter_key: str | None = None,
                 extraction_prompt: str | None = None,
                 input_text_key: str | None = None,
                 entity_types_key: str | None = None,
                 completion_delimiter_key: str | None = None,
                 max_gleanings: int | None = None,
                 ):
        self.llm = llm
        self.entity_types = entity_types or "entity_types"
        self.extraction_prompt = GRAPH_EXTRACTION_PROMPT
        self.repeat_prompt = REPEAT_PROMPT
        self._input_text_key = input_text_key or "input_text"
        self._tuple_delimiter_key = tuple_delimiter_key or "tuple_delimiter"
        self._record_delimiter_key = record_delimiter_key or "record_delimiter"
        self._completion_delimiter_key = (
            completion_delimiter_key or "completion_delimiter"
        )

        self._max_gleanings = max_gleanings

    def entity_extract(self, texts: list[str], event_ids: list[str]):
        processes: dict[str, list[str]] = {
            "event_id":[],
            "log_template":[],
            "processed_template":[],
            "summary": [],
            "source_entity":[],
            "source_entity_type":[],
            "target_entity":[],
            "target_entity_type":[],
            "action": [],
            "status": [],
            "java_exception":[]

        }

        prompt_variables = {}
        # Wire defaults into the prompt variables
        prompt_variables = {
            **prompt_variables,
            self._tuple_delimiter_key: prompt_variables.get(self._tuple_delimiter_key)
            or DEFAULT_TUPLE_DELIMITER,
            self._record_delimiter_key: prompt_variables.get(self._record_delimiter_key)
            or DEFAULT_RECORD_DELIMITER,
            self._completion_delimiter_key: prompt_variables.get(
                self._completion_delimiter_key
            )
            or DEFAULT_COMPLETION_DELIMITER,
            self.entity_types: ",".join(
                 DEFAULT_ENTITY_TYPES
            ),
        }

        for text, e_id in zip(texts, event_ids):
                raw_text = text
                text, java_exception = self._split_java_exception(text)
                text = self._remove_invalid_characters(text)

                # Invoke the entity extraction
                result = self.llm( GRAPH_EXTRACTION_PROMPT, variables={**prompt_variables, self._input_text_key: text}, if_extract=True)

                print(f"=== Log template {e_id}: ")
                print(text)
                print(f"==== LLM Raw Output {e_id}: ")
                print(result.output)
                try:
                    process_dict = result.to_json()
                except Exception as e:
                    logging.exception("Error parsing json! repeating...")
                    print(result.output)
                    result = self.llm(REPEAT_PROMPT)
                    print("==== LLM Repeat Output: ")
                    print(result.output)
                    process_dict = result.to_json()

                processes["event_id"].append(e_id)
                processes["log_template"].append(raw_text)
                processes["processed_template"].append(text)
                for (k, v) in process_dict.items():
                    if k in ['target_entity', 'source_entity']:
                        v = v.replace("_", "")
                    processes[k].append(v)
                processes["java_exception"].append(java_exception)


        structured_processes = pd.DataFrame(processes)
        cols = ['event_id', "log_template"] + [col for col in structured_processes.columns if
                                               col not in ['event_id', 'log_template']]
        structured_processes = structured_processes.reindex(columns=cols)
        return structured_processes

    def merge_target_and_source_entity(self, structured_processes):
        all_entities = set(structured_processes['source_entity'].unique()) | set(structured_processes['target_entity'].unique())
        all_entities.remove("none")
        if "<*>" in all_entities:
            all_entities.remove("<*>")
        entity_count = {
            e: len(structured_processes[structured_processes["source_entity"] == e]) + len(structured_processes[structured_processes["target_entity"] == e]) for
            e in all_entities}
        merged_entities = []
        merged_entity_types = []

        for id, row in structured_processes.iterrows():
            source = row["source_entity"]
            target = row["target_entity"]
            source_entity_count = entity_count[source] if source in entity_count.keys() else 0
            target_entity_count = entity_count[target] if target in entity_count.keys() else 0
            if source_entity_count > target_entity_count:
                entity = row["source_entity"]
                entity_type = row["source_entity_type"]
            else:
                entity = row["target_entity"]
                entity_type = row["target_entity_type"]

            merged_entities.append(entity)
            merged_entity_types.append(entity_type)

        return merged_entities, merged_entity_types

    def entity_filling(self, structured_processes):
        entity_list = structured_processes["entity"].unique().tolist()
        if 'none' in entity_list:
            entity_list.remove("none")
        entity_list_str = ','.join(entity_list)

        none_entities = structured_processes[structured_processes["entity"] == 'none']
        refined_entities = {}
        for id, row in none_entities.iterrows():
            print("==== Log template: ")
            print(row["log_template"])
            result = self.llm(ENTITY_FILLING_PROMPT, variables={"entity_types": DEFAULT_ENTITY_TYPES,"entity_list": entity_list_str, "input_text": row["log_template"]})
            print("==== Prompt ===")
            print(self.llm.prompt)
            print("==== LLM Raw Output: ")
            print(result.output)
            result_dict = result.to_json()
            entity = result_dict["entity"]
            refined_entities[id] = entity

        new_entities = []
        for i in range(0, len(structured_processes)):
            if i not in refined_entities.keys():
                entity = structured_processes.iloc[i]["entity"]
            else:
                entity = refined_entities[i]
            new_entities.append(entity)

        return new_entities

    def generate_unique_entity_name_and_id(self, entity_col, structured_processes):
        '''
        each entity is unique, and give none entity a unique name
        :param entity_col:
        :param structured_processes:
        :return:
        '''
        entities = structured_processes[entity_col].tolist()
        # unique entity for "none" with entity_id
        none_count = 0
        unique_entities = []
        for entity in entities:
            if entity == 'none':
                u_entity = "none" + str(none_count)
                none_count += 1
            else:
                u_entity = entity
            unique_entities.append(u_entity)

        unique_entity_set = set(unique_entities)
        entity_to_id = dict(zip(unique_entity_set, range(len(unique_entity_set))))
        entity_ids = [entity_to_id[u_e] for u_e in unique_entities]
        return unique_entities, entity_ids

    def generate_unique_action_name_and_id(self, entity_col, action_col, structured_processes: pd.DataFrame):
        '''
        each action is unique, and give none action a unique name
        :param entity_col:
        :param action_col:
        :param structured_processes:
        :return:
        '''
        none_count = 0
        unique_actions = []
        for i, row in structured_processes.iterrows():
            if row[action_col] != 'none':
                unique_action_name = row[entity_col] +','+ row[action_col]
            else:
                unique_action_name = row[entity_col]+','+ row[action_col]+"_"+ str(none_count)
                none_count += 1

            unique_actions.append(unique_action_name)

        unique_action_set = set(unique_actions)
        action_to_id = dict(zip(unique_action_set, range(len(unique_action_set))))
        action_ids = [action_to_id[u_a] for u_a in unique_actions]
        final_action_names = [u_a.split(",")[1] for u_a in unique_actions]
        return final_action_names, action_ids



    def action_refilling(self, structured_processes, entity_col ='entity_1'):

        entity_list = structured_processes[entity_col].unique().tolist()
        if 'none' in entity_list:
            entity_list.remove("none")
        refined_actions = {}

        for entity in entity_list:
            entity_processes = structured_processes[structured_processes[entity_col] == entity]
            valid_action_list = entity_processes["action"].unique().tolist()
            if 'none' in valid_action_list:
                valid_action_list.remove("none")
            if 'none' not in entity_processes["action"].tolist() or len(valid_action_list) == 0:
                continue
            else:
                none_actions = entity_processes[entity_processes["action"] =='none']
                for id, row in none_actions.iterrows():
                    print("==== Log template: ")
                    print(row["log_template"])
                    result = self.llm(ACTION_FILLING_PROMPT,
                                      variables={"action_list": valid_action_list, "input_text": row["log_template"],
                                                 "entity": entity})
                    print("==== LLM Raw Output: ")
                    print(result.output)
                    result_dict = result.to_json()
                    action = result_dict["action"]
                    refined_actions[id] = action
        new_actions = []
        for i in range(0, len(structured_processes)):
            if i not in refined_actions.keys():
                action = structured_processes.iloc[i]["action"]
            else:
                action = refined_actions[i]

            new_actions.append(action)
        return new_actions

    def _remove_invalid_characters(self, text: str) -> str:
        # Use regular expression to remove substrings enclosed in <*>
        cleaned_text = re.sub(r"<\*>", "", text)
        return cleaned_text

    def _split_java_exception(self, text: str, ):
        # remove java exception information
        index = text.find('java.')
        java_exception = ""
        # If 'java.' is found, slice the string up to that index
        if index != -1:
            java_exception = text[index:]
            return text[:index], java_exception

        # If 'java.' is not found, return the original string
        return text, java_exception


if __name__ == '__main__':
    llm = LLM(model="gpt-3.5-turbo")
    dataset = 'ThunderBird'
    extractor = Extractor(llm)
    ############## Extraction ##############
    template_df = pd.read_csv(f"../data/{dataset}/{dataset}.log_templates.csv")
    templates = template_df["EventTemplate"].tolist()
    template_krone_tree = extractor.entity_extract(texts=templates, event_ids=template_df["EventId"])
    first_entities, entity_types = extractor.merge_target_and_source_entity(template_krone_tree)
    template_krone_tree["entity"] = first_entities
    template_krone_tree["entity_type"] = entity_types
    template_krone_tree = template_krone_tree.sort_values(by = ["entity"])
    print(template_krone_tree[["entity", "action", "status"]])


    ############## Refinement: Action Candidate Selection ##############
    print("******************** Start Entity Refilling ********************")
    # structured_processes = pd.read_csv(f"../output/{dataset}/templates_krone_tree.csv")
    entities_1 = extractor.entity_filling(template_krone_tree)
    template_krone_tree["entity_1"] = entities_1


    ############## Refinement: Action Candidate Selection ##############
    # structured_processes = pd.read_csv(f"../output/{dataset}/templates_krone_tree.csv")

    print("******************** Start Action Refilling ********************")
    actions_1 = extractor.action_refilling(template_krone_tree)
    template_krone_tree["action_1"] = actions_1
    template_krone_tree["status_id"] = template_krone_tree.index.values

    ############## CamelCase ##############
    # structured_processes = pd.read_csv(f"../output/{dataset}/templates_krone_tree.csv")

    template_krone_tree["action_1"] = template_krone_tree["action_1"].map(to_camel_case)
    template_krone_tree["entity_1"] = template_krone_tree["entity_1"].map(to_camel_case)

    if not os.path.exists(f"../output/{dataset}/"):
        os.makedirs(f"../output/{dataset}/")
    template_krone_tree.to_csv(f"../output/{dataset}/templates_krone_tree.csv", index=False)
