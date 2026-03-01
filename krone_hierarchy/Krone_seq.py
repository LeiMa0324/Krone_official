import ast
import time

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from llm.llm import *
from krone_hierarchy.Node import Node
from tqdm import tqdm
from krone_hierarchy.PROMPTS import *
DELIMITER = '|'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import LongformerModel, LongformerTokenizer
import torch


INITIAL_BLANK_SUMMARY = ''

LOCAL_DETECTOR_NORMAL_REASON ='LOCAL_DETECTOR_NORMAL-NONGT'
LOCAL_DETECTOR_ABNORMAL_REASON ='LOCAL_DETECTOR_ABNORMAL-NONGT'
TRAIN_NORMAL_REASON ='GT'
EMPTY_PATH_NORMAL_REASON ='EMPTY_PATH_NORMAL'

DEFAULT_PRED = -1



def is_sublist(test, trains):
    test_seq = '|'.join(test)
    for train in trains:
        train_seq = '|'.join(train)
        if test_seq in train_seq:
            return True
    return False


def find_largest_prefix_subsequence(test, train_seqs):
    if len(test) <= 1:
        return [], (0, 0)
    largest_prefix_subsequence = []
    for i in range(0, len(test)):
        subseq = test[: i+1]
        is_sub = is_sublist(subseq, train_seqs)
        if is_sub:
            largest_prefix_subsequence.append(test[i])
        else:
            return largest_prefix_subsequence, (0, i)
    return largest_prefix_subsequence, (0, len(test))

def find_largest_suffix_subsequence(test, train_seqs):
    if len(test) <= 1:
        return [], (len(test), len(test))
    largest_suffix_subsequence = []
    for i in range(len(test)-1, -1, -1):
        subseq = test[i: len(test)]
        is_sub = is_sublist(subseq, train_seqs)
        if is_sub:
            largest_suffix_subsequence.append(test[i])
        else:
            largest_suffix_subsequence.reverse()
            return largest_suffix_subsequence, (i +1,len(test))
    largest_suffix_subsequence.reverse()
    return largest_suffix_subsequence, (0, len(test))


def contains_existing_seq(test_seq, train_seqs, ):
    direct_sub =  is_sublist(test_seq, train_seqs)
    if direct_sub:
        return True
    else:
        largest_prefix_subsequence, prefix_range = find_largest_prefix_subsequence(test_seq, train_seqs)
        largest_suffix_subsequence, suffix_range = find_largest_suffix_subsequence(test_seq, train_seqs)
        if prefix_range[1] > suffix_range[0]:
            return True
        else:
            while prefix_range[1]-prefix_range[0]> 0 and suffix_range[1]-suffix_range[0]> 0 and prefix_range[1] < suffix_range[0]+1:
                test_seq = test_seq[prefix_range[1]:suffix_range[0]+1]
                largest_prefix_subsequence, prefix_range = find_largest_prefix_subsequence(test_seq, train_seqs)
                largest_suffix_subsequence, suffix_range = find_largest_suffix_subsequence(test_seq, train_seqs)
                if prefix_range[1] > suffix_range[0]:
                    return True

            return False


def condense_path(node_path: list['Node']) -> list['Node']:
    '''remove the consecutive repeating events in the sequence, except for the status'''
    # open = entity_path[0].node_type not in ['STATUS']
    open = node_path[0].node_type not in ['STATUS']
    if not open:
        return node_path
    curr_entity_node = node_path[0]
    condensed_entity_path = [curr_entity_node]
    for entity in node_path:
        if entity == curr_entity_node:
            continue
        else:
            condensed_entity_path.append(entity)
            curr_entity_node = entity
    return condensed_entity_path

class KroneSeq:
    def __init__(self, node_list: List[Node], seq_id:int, parent_node: Node, children_paths:List['KroneSeq'] = None,
                 summary = INITIAL_BLANK_SUMMARY, pred: int = DEFAULT_PRED, reason: str= '', if_GT = False):
        self.node_list: List[Node] = node_list
        self.uncollapsed_no_list: List[Node] = node_list

        if if_GT:
            self.connect_nodes() # only connect nodes for gt paths
        self.children_paths: List['KroneSeq'] = children_paths
        self.level = self.node_list[0].node_type
        self.parent_node = parent_node
        self.log_keys= []


        self.entity_identifier = ''
        self.action_identifier = ''
        self.status_identifier = ''
        self.overall_identifier = ''

        self.semantic_entity_identifier = ''
        self.semantic_action_identifier = ''
        self.semantic_status_identifier = ''

        self.path_summary = summary
        self.path_pred = pred
        self.path_pred_reason = reason
        self.llm_predicted = False

        self.DELIMITER = '|'

        # embedding
        self.summary_embedding = None
        self.pattern_embedding = None
        self.pattern_seq = ''

        self.sequence_ids = set()
        if seq_id != -1:
            self.sequence_ids.add(seq_id)

        self.entity_identifier, self.semantic_entity_identifier, self.action_identifier, self.semantic_action_identifier,\
         self.status_identifier, self.semantic_status_identifier, self.overall_identifier = self._generate_path_identifier()

        self.is_empty_path = False
        self.get_logkeys()
        self.set_empty_masked_path()

    def get_logkeys(self):
        if self.log_keys == []:
            nodes = self.find_all_status_nodes()
            assert len(nodes) == len(self.status_identifier.split(','))
            self.log_keys = [str(next(iter(node.template_ids))) for node in nodes]

        return self.log_keys

    def connect_nodes(self):
        prev_node = None
        for i in range(0, len(self.node_list)):
            curr_node = self.node_list[i]
            if i < len(self.node_list) - 1:
                next_node = self.node_list[i+1]
                curr_node.outgoing_neighbors[next_node.node_identifier] = next_node
            if i > 0:
                curr_node.incoming_neighbors[prev_node.node_identifier] = prev_node
            prev_node = curr_node

    def _find_empty_masked_path(self):

        if self.level =='STATUS':
            for node in self.node_list:
                if not node.masked:
                    return False
            return True
        else:
            non_empty_c_paths = self.get_non_empty_children_paths()
            return len(non_empty_c_paths) == 0

    def set_empty_masked_path(self):
        self.is_empty_path = self._find_empty_masked_path()

    def find_all_status_nodes(self, ):

        if self.level =='STATUS':
            return self.node_list
        else:
            nodes = []
            for child_path in self.children_paths:
                nodes.extend(child_path.find_all_status_nodes())
            return nodes

    def find_logkey_sequence_str(self):

        if self.pattern_seq == '':
            # nodes = self.find_all_status_nodes()
            # assert len(nodes) == len(self.status_identifier.split(','))
            # self.log_keys = [str(next(iter(node.template_ids))) for node in nodes]
            pattern_seq = ' '.join(self.log_keys)
            self.pattern_seq = pattern_seq
            return pattern_seq

        else:
            return self.pattern_seq

    def generated_pattern_embedding(self):

        if self.pattern_embedding is None and (not self.is_empty_path):
            # self.pattern_embedding = torch.randn(768)
            printed = True
            sequence = self.find_logkey_sequence_str()
            print(f"     EMBEDDING GENERATION.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            model = model.to(device)

            # Tokenize the input for the batch
            transformer_model = model._first_module().auto_model
            inputs = model.tokenizer(sequence, padding=True, truncation=True, return_tensors="pt")

            # Generate embeddings for the batch
            with torch.no_grad():
                outputs = transformer_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                            return_dict=True)
                # Get the embeddings for the batch (last hidden state of the [CLS] token for each sequence)
                self.pattern_embedding = outputs.last_hidden_state[0:, 0,
                                             :]  # Select [CLS] token embeddings for all sequences


        return self.pattern_embedding

    def get_non_empty_children_paths(self) -> List['KroneSeq']:
        non_empty_children_paths = []
        for c_path in self.children_paths:
            if not c_path.is_empty_path:
                non_empty_children_paths.append(c_path)

        return non_empty_children_paths

    def add_sequence_id(self, seq_id: int):
        self.sequence_ids.add(seq_id)

    def has_path_pred(self):
        return self.path_pred != DEFAULT_PRED

    def has_path_reason(self):
        return self.path_pred_reason != ''

    def get_path_summary(self):
        return self.path_summary

    def get_path_pred(self):
        return self.path_pred

    def get_path_pred_reason(self):
        return self.path_pred_reason

    def _generate_path_identifier(self):
        DELIMITER = '|'

        if self.level == 'STATUS':
            status_identifier, semantic_status_identifier = self.generate_status_identifier()
            action_identifier, semantic_action_identifier = self.generate_action_identifier()
            entity_identifier, semantic_entity_identifier = self.generate_entity_identifier()

        elif self.level == 'ACTION':
            status_identifier, semantic_status_identifier = self._concate_status_identifier()
            action_identifier, semantic_action_identifier = self.generate_action_identifier()
            entity_identifier, semantic_entity_identifier  = self.generate_entity_identifier()

        elif self.level == 'ENTITY':
            status_identifier, semantic_status_identifier = self._concate_status_identifier()
            action_identifier, semantic_action_identifier = self._concate_action_identifier()
            entity_identifier, semantic_entity_identifier  = self.generate_entity_identifier()

        else:
            raise NotImplementedError


        overall_identifier = 'ENTITY: '+ entity_identifier+DELIMITER+"ACTION: "+action_identifier+DELIMITER+"STATUS: "+status_identifier

        return entity_identifier, semantic_entity_identifier, action_identifier,semantic_action_identifier, \
                status_identifier,semantic_status_identifier, \
                overall_identifier

    def generate_status_identifier(self):
        if self.level == 'STATUS':
            status_names = []  # find action identifier
            semantic_status_names = []
            for node in self.node_list:
                status_names.append(node.node_identifier)
                semantic_status_names.append(node.node_identifier.split("_")[0])
            status_identifier = ','.join(status_names)
            semantic_status_identifier = ','.join(semantic_status_names)
        else:
            raise NotImplementedError(f"{self.level} level not supported for _generate_status_identifier!")
        return status_identifier, semantic_status_identifier

    def _concate_status_identifier(self):
        if self.level in ['ACTION', 'ENTITY']:
            status_names = []  # find action identifier
            semantic_status_names = []  # find action identifier
            for c_path in self.children_paths:
                if self.level == 'ACTION':
                    s_name = c_path.status_identifier
                    semantic_s_name = c_path.status_identifier.split("_")[0]
                else:
                    s_name = ','.join([grand_c_path.status_identifier for grand_c_path in c_path.children_paths])
                    semantic_s_name = ','.join([grand_c_path.status_identifier.split("_")[0] for grand_c_path in c_path.children_paths])

                status_names.append(s_name)
                semantic_status_names.append(semantic_s_name)
            status_identifier = ','.join(status_names)
            semantic_status_identifier = ','.join(semantic_status_names)
        else:
            raise NotImplementedError(f"{self.level} level not supported for _concate_status_identifier!")
        return status_identifier, semantic_status_identifier

    def _concate_action_identifier(self):
        if self.level in ['ENTITY']:
            action_names = []  # find action identifier
            semantic_action_names = []
            for c_path in self.children_paths:
                a_name = c_path.action_identifier
                semantic_a_name = c_path.semantic_action_identifier
                action_names.append(a_name)
                semantic_action_names.append(semantic_a_name)
            action_identifier = ','.join(action_names)
            semantic_action_identifier = ','.join(semantic_action_names)
        else:
            raise NotImplementedError(f"{self.level} level not supported for _concate_action_identifier!")
        return action_identifier, semantic_action_identifier

    def generate_action_identifier(self):
        if self.level in['STATUS', 'ACTION']:
            action_names = [self.node_list[0].node_identifier] if self.level == 'ACTION' else [
                self.node_list[0].parent.node_identifier]
            semantic_action_names = [self.node_list[0].node_identifier.split("_")[0]] if self.level == 'ACTION' else [
                self.node_list[0].parent.node_identifier.split("_")[0]]

            for node in self.node_list:
                if self.level == 'ACTION':
                    action_name = node.node_identifier
                    semantic_action_name = node.node_identifier.split("_")[0]
                else:
                    action_name = node.parent.node_identifier
                    semantic_action_name = node.parent.node_identifier.split("_")[0]

                if action_name != action_names[-1]:
                    action_names.append(action_name)
                    semantic_action_names.append(semantic_action_name)
                else:
                    continue

            action_identifier = ','.join(action_names)
            semantic_action_identifier = ','.join(semantic_action_names)

        else:
            raise NotImplementedError(f"{self.level} level not supported for _generate_action_identifier!")
        return action_identifier, semantic_action_identifier

    def generate_entity_identifier(self):
        if self.level in['STATUS', 'ACTION', 'ENTITY']:
            entity_names = [self.node_list[0].find_entity_node().node_identifier]
            semantic_entity_names = [self.node_list[0].find_entity_node().node_identifier.split("_")[0]]
            for node in self.node_list:
                entity_node = node.find_entity_node()
                if entity_node.node_identifier != entity_names[-1]:
                    entity_names.append(entity_node.node_identifier)
                    semantic_entity_names.append(entity_node.node_identifier.split("_")[0])
            entity_identifier = ','.join(entity_names)
            semantic_entity_identifier = ','.join(semantic_entity_names)
        else:
            raise NotImplementedError(f"{self.level} level not supported for _generate_action_identifier!")
        return entity_identifier, semantic_entity_identifier

    def generated_process_sequence_text(self) -> str:
        path_seqs = []
        if self.level !='STATUS':
            non_empty_children_paths = self.get_non_empty_children_paths()
            if len(non_empty_children_paths) > 0:
                for i, child in enumerate(non_empty_children_paths):
                    if child.path_summary == INITIAL_BLANK_SUMMARY:
                        print("Empty summary path!")
                        print(child.level)
                        print(child.overall_identifier)
                        print(child.path_pred)
                        print(child.path_pred_reason)
                    assert child.path_summary != INITIAL_BLANK_SUMMARY # 默认所有的path都有summary
                    path_seqs.append(f"{i}: "+ child.path_summary.replace('"', "'"))

        else:
            unmasked_nodes = []
            for node in self.node_list:
                if not node.masked:
                    unmasked_nodes.append(node)
            for i, node in enumerate(unmasked_nodes):
                if not node.masked:
                    path_seqs.append(f"{i}: "+ (node.template_summary))
        path_seq_text = DELIMITER.join(path_seqs)
        return path_seq_text

    def dummy_summarize_path(self, llm: LLM):
        if self.level =='STATUS' and len(self.node_list) == 1:  # the 1-length path summary == single template summary
            print("Status path with length 1, skipping LLM summarization.")
            summary = self.node_list[0].template_summary
            assert summary != INITIAL_BLANK_SUMMARY

        else:  # ask llm to summarize for length > 1 path
            summary = 'DUMMY SUMMARTY'
        self.path_summary = summary
        return summary

    def summarize_path(self, llm: LLM, hardcode_kleene_pattern_summary = False):
        llm_summary = 0

        if self.level =='STATUS' and len(self.node_list) == 1:  # the 1-length path summary == single template summary
            print("Status path with length 1, skipping LLM summarization.")
            summary = self.node_list[0].template_summary
            self.path_summary = summary
            assert summary != INITIAL_BLANK_SUMMARY
            return summary, llm_summary

        else:  # ask llm to summarize for length > 1 path
            if hardcode_kleene_pattern_summary and self.level == 'STATUS' and len(self.node_list) > 1 and len(
                    set(self.node_list)) == 1:  # kleene pattern summary
                print("hard coded summary for kleene status path")
                summary = (
                    f"The system executed a sequence of the same task: '{self.node_list[0].template_summary}' in each step. "
                    f"The task was repeated for {len(self.node_list)} times.")
                self.path_summary = summary
                return summary, llm_summary

            if self.children_paths is not None and len(self.children_paths) == 1:
                summary = self.children_paths[0].path_summary
                self.path_summary = summary
                return summary, llm_summary
            else:

                input_process_seq = self.generated_process_sequence_text()
                assert input_process_seq and (not input_process_seq.isspace())

                prompt = PROCESS_SEQ_SUMMARY_PROMPT_V2
                varibales = {
                    "input_process_seq": input_process_seq,
                    "DELIMITER": DELIMITER,
                }

                prompt = self.format_prompt(prompt, prompt_variables=varibales)
                # print(f"==== {self.level} path summary prompt ====")

                call_detail = {"call_type": 'summary',
                               "path_level": self.level,
                               "path_identifier": self.overall_identifier,
                               "logkey_seq": self.find_logkey_sequence_str(),
                               "seq_ids": list(self.sequence_ids),
                               "example_seq_ids": "",
                               "example_pattern_num": 0
                               }

                # print(prompt)
                response = llm(prompt = prompt, call_detail=call_detail)
                summary = response.output
                llm_summary+=1

                self.path_summary = summary
                return summary,llm_summary

    def format_prompt(self, prompt, prompt_variables):
        for key, val in prompt_variables.items():
            prompt = prompt.replace("{" + f"{key}" + "}", f'{val}')
        return prompt

    def find_log_key_seq(self) -> List[str]:

        if self.level == 'STATUS':
            return [next(iter(node.template_ids)) for node in self.node_list]
        else:
            logkey_seqs = []
            for c_path in self.children_paths:
                logkey_seqs.extend(c_path.find_log_key_seq())
            return logkey_seqs


