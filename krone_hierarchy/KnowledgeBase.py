from numpy.distutils.conv_template import header

from krone_hierarchy.Node import *
from llm.llm import *
import pickle
from utils import *
import time
import ast
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from krone_hierarchy.utils import *
from tqdm import tqdm
from krone_hierarchy.Krone_seq import *
from krone_hierarchy.Krone_seq_manager import *
from  executor.time_tracker import TimeTracker


DELIMITER = '|'
STAGES = ['0 - INITIALIZED', '1 - GT PATH SUMMARIZATION', '2 - STATUS PATH DETECTION',
'3 - ACTION PATH DETECTION','4 - ENTITY PATH DETECTION']


class KnowledgeBase:
    def __init__(self, llm: LLM,time_tracker: TimeTracker, hardcode_kleene_pattern_summary= False):
        self.llm = llm
        self.hardcode_kleene_pattern_summary = hardcode_kleene_pattern_summary
        # the summary for known path in the training set
        self.GT_status_path_manager = Path_manager(level ='STATUS', if_GT=True, llm=llm)
        self.GT_action_path_manager = Path_manager(level ='ACTION', if_GT=True, llm=llm)
        self.GT_entity_path_manager = Path_manager(level ='ENTITY', if_GT=True, llm=llm)

        self.non_GT_status_path_manager = Path_manager(level ='STATUS', if_GT=False, llm=llm)
        self.non_GT_action_path_manager = Path_manager(level ='ACTION', if_GT=False, llm=llm)
        self.non_GT_entity_path_manager = Path_manager(level ='ENTITY', if_GT=False, llm=llm)

        self.stage = STAGES[0]
        self.dummy_llm_detection_calls = 0
        self.dummy_llm_summary_calls = 0

        self.diff_entity_path_segments: Dict[str: KroneSeq] = {}
        self.seq_num_with_diffs = 0
        self.automaton_fixed = 0

        self.seq_detected_by_nodes = set()


        self.action_deeplog_detectors = {} # learning status transition for an action
        self.entity_deeplog_detectors = {}
        self.root_deeplog_detectors = {}
        self.time_tracker = time_tracker


    def create_paths_for_sequence(self,  entity_nodes: List['Node'], action_nodes: List['Node'], status_nodes:  List[Node], seq_id, if_GT):


        status_paths, status_path_idx = self._create_status_paths(entity_nodes, action_nodes, status_nodes, seq_id,
                                                                  if_GT)

        action_paths = self._create_action_paths(entity_nodes, action_nodes, status_paths, status_path_idx, seq_id, if_GT)
        entity_path = self._create_entity_paths(entity_nodes, action_paths, seq_id, if_GT)

        return entity_path, action_paths, status_paths

    def _create_status_paths(self, entity_nodes, action_nodes, status_nodes, seq_id, if_GT):
        last_action = action_nodes[0]
        curr_action_statuses = []
        status_paths = []
        status_path_idx: List[Tuple[int, int]] = []  # the same size of the action nodes
        start = 0
        end = 0
        i = 0
        for entity, action, status in zip(entity_nodes, action_nodes, status_nodes):
            end = i
            if action == last_action:
                curr_action_statuses.append(status)
            else:

                temp_status_path = self.GT_status_path_manager.generate_temp_path(node_list=curr_action_statuses,
                                                                                  seq_id=seq_id, if_GT= if_GT)
                if if_GT:
                    status_path = self.GT_status_path_manager.add_path(temp_status_path, seq_id=seq_id, maintain_automaton = True )
                    status_path.path_pred=0
                    status_path.path_pred_reason='GT'
                else:
                    if self.GT_status_path_manager.has_path(temp_status_path):
                        status_path = self.GT_status_path_manager.get_path(temp_status_path.overall_identifier)
                        del temp_status_path
                    else:
                        status_path = self.non_GT_status_path_manager.add_path(temp_status_path, seq_id=seq_id)

                status_path_idx.append((start, end))
                curr_action_statuses = [status]
                status_paths.append(status_path)
                start = i

            last_action = action
            i += 1

        if len(curr_action_statuses) > 0:
            temp_status_path = self.GT_status_path_manager.generate_temp_path(node_list=curr_action_statuses,
                                                                              seq_id=seq_id, if_GT= if_GT)
            if if_GT:
                status_path = self.GT_status_path_manager.add_path(temp_status_path, seq_id=seq_id, maintain_automaton = True )
                status_path.path_pred = 0
                status_path.path_pred_reason = 'GT'
            else:
                if self.GT_status_path_manager.has_path(temp_status_path):
                    status_path = self.GT_status_path_manager.get_path(temp_status_path.overall_identifier)
                    del temp_status_path
                else:
                    status_path = self.non_GT_status_path_manager.add_path(temp_status_path, seq_id=seq_id)

            status_paths.append(status_path)
            status_path_idx.append((start, end+1))

        time3 = time.time()
        return status_paths, status_path_idx

    def _create_action_paths(self, entity_nodes, action_nodes, status_paths, status_path_idx, seq_id, if_GT):
        action_paths = []
        curr_entity_actions = []
        curr_start = 0
        curr_end = 0
        i = 0
        last_entity = entity_nodes[0]
        for entity, action, in zip(entity_nodes, action_nodes):
            curr_end = i
            if entity == last_entity:
                curr_entity_actions.append(action)
            else:

                s_path_start_idx = 0
                s_path_end_idx = 0
                for k, idx_tupe in enumerate(status_path_idx):
                    if idx_tupe[0] == curr_start:
                        s_path_start_idx = k
                    if idx_tupe[1] == curr_end:
                        s_path_end_idx = k + 1
                        break

                action_path_status_paths = status_paths[
                                           s_path_start_idx:s_path_end_idx]  # maintain action path's children paths

                temp_action_path = self.GT_action_path_manager.generate_temp_path(node_list=curr_entity_actions,
                                                                                  children_paths=action_path_status_paths,
                                                                                  seq_id=seq_id, if_GT= if_GT)
                if if_GT:
                    action_path = self.GT_action_path_manager.add_path(temp_action_path, seq_id=seq_id, maintain_automaton = True  )
                    action_path.path_pred=0
                    action_path.path_pred_reason='GT'
                else:
                    if self.GT_action_path_manager.has_path(temp_action_path):
                        action_path = self.GT_action_path_manager.get_path(temp_action_path.overall_identifier)
                        del temp_action_path
                    else:
                        action_path = self.non_GT_action_path_manager.add_path(temp_action_path, seq_id=seq_id)

                action_paths.append(action_path)
                curr_start = i
                curr_entity_actions = [action]
            last_entity = entity
            i += 1
        curr_end+=1

        if len(curr_entity_actions) > 0:
            s_path_start_idx = 0
            s_path_end_idx = 0
            for j, idx_tupe in enumerate(status_path_idx):
                if idx_tupe[0] == curr_start:
                    s_path_start_idx = j
                if idx_tupe[1] == curr_end:
                    s_path_end_idx = j + 1
                    break
            action_path_status_paths = status_paths[
                                       s_path_start_idx:s_path_end_idx]  # maintain action path's children paths
            temp_action_path = self.GT_action_path_manager.generate_temp_path(node_list=curr_entity_actions,
                                                                              children_paths=action_path_status_paths,
                                                                              seq_id=seq_id, if_GT= if_GT)
            if if_GT:
                action_path = self.GT_action_path_manager.add_path(temp_action_path, seq_id=seq_id, maintain_automaton = True )
                action_path.path_pred = 0
                action_path.path_pred_reason = 'GT'
            else:
                if self.GT_action_path_manager.has_path(temp_action_path):
                    action_path = self.GT_action_path_manager.get_path(temp_action_path.overall_identifier)
                    del temp_action_path
                else:
                    action_path = self.non_GT_action_path_manager.add_path(temp_action_path, seq_id=seq_id)
            action_paths.append(action_path)
        return action_paths

    def _create_entity_paths(self, entity_nodes, action_paths, seq_id, if_GT):
        temp_entity_path = self.GT_entity_path_manager.generate_temp_path(node_list=entity_nodes,
                                                                          children_paths=action_paths,
                                                                          seq_id=seq_id, if_GT= if_GT)
        if if_GT:
            entity_path = self.GT_entity_path_manager.add_path(temp_entity_path, seq_id=seq_id, maintain_automaton = True)
              # maintain entity level automaton
            entity_path.path_pred = 0
            entity_path.path_pred_reason = 'GT'
        else:
            if self.GT_entity_path_manager.has_path(temp_entity_path):
                entity_path = self.GT_entity_path_manager.get_path(temp_entity_path.overall_identifier)
                del temp_entity_path
            else:
                entity_path = self.non_GT_entity_path_manager.add_path(temp_entity_path, seq_id=seq_id)

        return entity_path

    def general_sequence_detect_v2(self, test_path: KroneSeq, seq_id, k, detect_mode, example_source='GT', edge_consecutive_sensitive = False, automaton_adjustment= False,
                                   dummy_summarize = False, dummy_detect = False):

        llm_call = 0
        local_pred, if_true_GT_path = self.general_sequence_detect_local(test_path, seq_id, automaton_adjustment= automaton_adjustment, edge_consecutive_sensitive= edge_consecutive_sensitive)

        llm_pred = local_pred
        if detect_mode =='local':
            if not if_true_GT_path:
                path_reason = LOCAL_DETECTOR_NORMAL_REASON if local_pred ==0 else LOCAL_DETECTOR_ABNORMAL_REASON
                test_path.path_pred = llm_pred
                test_path.path_pred_reason = path_reason
            return local_pred, local_pred, llm_call
        else:   # llm and knowledge mode

            # 3. local normal, directly return
            if local_pred == 0:
                if detect_mode == 'llm' and (not if_true_GT_path): # summarize for non-GT normal path
                    if test_path.path_summary == INITIAL_BLANK_SUMMARY and test_path.level != 'ENTITY':
                        print(f'     ------ {test_path.level} Pattern: {test_path.find_log_key_seq()} ------')
                        # 1. load summary
                        if test_path.level == 'STATUS':
                            path_summary = self.non_GT_status_path_manager.load_path_summary_for_path(
                                test_path)
                        else:
                            path_summary = self.non_GT_action_path_manager.load_path_summary_for_path(
                                test_path)

                        if path_summary:
                            print(f"Loaded summary: {test_path.path_summary}")
                        else:
                            # summarize if not exist
                            if dummy_summarize:
                                test_path.dummy_summarize_path(self.llm)
                                self.dummy_llm_summary_calls += 1
                            else:
                                test_path.summarize_path(self.llm, self.hardcode_kleene_pattern_summary)
                            print(f"Generate summary: {test_path.path_summary}")

                return local_pred, llm_pred, llm_call
            else:  # local anomaly
                # 1. load the test path knowledge for both normal and abnormal test path
                if detect_mode =='llm':
                    self.load_everything_for_path(test_path)

                ## 2. check existing knowledge for both llm and knowledge mode
                curr_time = time.time()
                if test_path.has_path_pred():
                    llm_pred = int(test_path.path_pred)
                    self.time_tracker.update_knowledge_test(time.time()-curr_time)
                    return local_pred, llm_pred, llm_call

                # 3. test for status nodes for both llm and knowledge mode to save llm requests
                if test_path.level == 'STATUS':
                    contain_anomaly_nodes = False
                    pred_reason = ''
                    for node in test_path.node_list:
                        if node.is_anomaly:
                            print(f"Abnormal node {node.template_ids}, skip LLM request")
                            contain_anomaly_nodes = True
                            pred_reason = node.is_anomaly_reason
                            break

                    if contain_anomaly_nodes:  # return prediction if contains any abnormal nodes
                        test_path.path_pred = 1
                        test_path.path_pred_reason = pred_reason
                        llm_pred = 1
                        self.seq_detected_by_nodes.add(seq_id)
                        return local_pred, llm_pred, llm_call

                # 4.llm requests
                if detect_mode =='llm':
                    # load summary
                    if test_path.path_summary == INITIAL_BLANK_SUMMARY and test_path.level != 'ENTITY':
                        print(f'     ------ {test_path.level} Pattern: {test_path.find_log_key_seq()} ------')
                        # 1. load summary
                        if test_path.level == 'STATUS':
                            path_summary = self.non_GT_status_path_manager.load_path_summary_for_path(
                                test_path)
                        else:
                            path_summary = self.non_GT_action_path_manager.load_path_summary_for_path(
                                test_path)
                        if path_summary:
                            print(f"Loaded summary: {test_path.path_summary}")
                        else:
                            # summarize if not exist
                            if dummy_summarize:
                                test_path.dummy_summarize_path(self.llm)
                                self.dummy_llm_summary_calls += 1
                            else:
                                test_path.summarize_path(self.llm, self.hardcode_kleene_pattern_summary)
                            print(f"Generate summary: {test_path.path_summary}")


                    # 3. use LLM to detect
                    if dummy_detect:
                        llm_pred = self.dummy_general_sequence_detect_llm(test_path)
                        self.dummy_llm_detection_calls +=1
                    else:
                        llm_pred = self.general_sequence_detect_llm(test_path,seq_id, k=k, example_source= example_source)
                    llm_call = 1
                    return local_pred, llm_pred, llm_call
                # 4. if nothing applies, return original local pred
                return local_pred, llm_pred, llm_call

    def dummy_general_sequence_detect_llm(self, test_path):
        test_path.path_pred = 1
        test_path.path_pred_reason = 'DUMMY DETECT'
        test_path.llm_predicted = True
        return 1

    def general_sequence_detect_local(self, test_path, seq_id, automaton_adjustment= False, edge_consecutive_sensitive=False):
        level = test_path.level

        if level == 'STATUS':
            curr_time = time.time()
            has_GT_path_cur_level = self.GT_status_path_manager.has_path(test_path, identifier_level=test_path.level)
            self.time_tracker.update_pattern_test(time.time()-curr_time)
            if not has_GT_path_cur_level:
                print(f"unmatched Status sequence: {test_path.entity_identifier}, finding path diffs")
                path_diffs_results = self.GT_status_path_manager.automaton_graph.path_diffs(path=test_path, edge_consecutive_sensitive=edge_consecutive_sensitive)
                if automaton_adjustment and len(path_diffs_results) ==0:
                    has_GT_path_cur_level = True  # i
                    self.automaton_fixed += 1

            true_GT_path = self.GT_status_path_manager.has_path(test_path)
            if true_GT_path:
                self.non_GT_status_path_manager.remove_path(test_path)  # remove path from non-gt, it is a true GT

        elif level == 'ACTION':
            curr_time = time.time()
            has_GT_path_cur_level = self.GT_action_path_manager.has_path(test_path, identifier_level=test_path.level)
            self.time_tracker.update_pattern_test(time.time()-curr_time)
            if not has_GT_path_cur_level:
                print(f"unmatched Action sequence: {test_path.entity_identifier}, finding path diffs")
                path_diffs_results = self.GT_action_path_manager.automaton_graph.path_diffs(path=test_path, edge_consecutive_sensitive=edge_consecutive_sensitive)
                if automaton_adjustment and len(path_diffs_results) ==0:
                    has_GT_path_cur_level = True  # i
                    self.automaton_fixed += 1
            true_GT_path = self.GT_action_path_manager.has_path(test_path)
            if true_GT_path:
                self.non_GT_action_path_manager.remove_path(test_path)

        elif level == 'ENTITY':
            curr_time = time.time()
            has_GT_path_cur_level = self.GT_entity_path_manager.has_path(test_path, identifier_level=test_path.level)
            self.time_tracker.update_pattern_test(time.time()-curr_time)
            if not has_GT_path_cur_level:
                print(f"unmatched ENTITY sequence: {test_path.entity_identifier}, finding path diffs")
                path_diffs_results = self.GT_entity_path_manager.automaton_graph.path_diffs(path=test_path, edge_consecutive_sensitive=edge_consecutive_sensitive)
                if automaton_adjustment and len(path_diffs_results) ==0:
                    has_GT_path_cur_level = True  # i
                    self.automaton_fixed += 1
                for (cur_diff_nodes, cur_diff_children_paths) in path_diffs_results:
                    diff_segment_path = self.GT_entity_path_manager.generate_temp_path(node_list=cur_diff_nodes,
                                                                                          children_paths=cur_diff_children_paths,
                                                                                          seq_id=seq_id, if_GT=False)
                    self.diff_entity_path_segments[diff_segment_path.overall_identifier] = diff_segment_path
                if len(path_diffs_results) > 0:
                    self.seq_num_with_diffs +=1

            true_GT_path = self.GT_entity_path_manager.has_path(test_path)

            if true_GT_path:
                self.non_GT_entity_path_manager.remove_path(test_path)
        else:
            raise NotImplementedError

        local_pred = 0 if has_GT_path_cur_level else 1
        return local_pred, true_GT_path

    def load_everything_for_path(self, test_path: KroneSeq):
        if test_path.level =='STATUS':
            self.non_GT_status_path_manager.load_path_embedding_for_path(
                test_path)  # if not, load the embedding to the path
            self.non_GT_status_path_manager.load_path_summary_for_path(
                test_path)  # if not, load the embedding to the path
            self.non_GT_status_path_manager.load_path_prediction_for_path(test_path)  # if not, load the knowledge to the path
        elif test_path.level =='ACTION':
            self.non_GT_action_path_manager.load_path_embedding_for_path(
                test_path)  # if not, load the embedding to the path
            self.non_GT_action_path_manager.load_path_summary_for_path(
                test_path)  # if not, load the embedding to the path
            self.non_GT_action_path_manager.load_path_prediction_for_path(test_path)  # if not, load the knowledge to the path
        elif test_path.level =='ENTITY':
            self.non_GT_entity_path_manager.load_path_embedding_for_path(
                test_path)  # if not, load the embedding to the path
            self.non_GT_entity_path_manager.load_path_prediction_for_path(test_path)  # if not, load the knowledge to the path
        else:
            raise NotImplementedError


    def general_sequence_detect_llm(self, test_path: KroneSeq, seq_id,
                                    k = 3, example_source ='mix'):
        level = test_path.level

        if_embedding = test_path.pattern_embedding is None and (
            not test_path.is_empty_path)
        if if_embedding:
            test_path.generated_pattern_embedding()
        if level == 'STATUS':
            gt_neighbor_paths, gt_scores = self.GT_status_path_manager.find_similar_paths_by_embedding(test_path, k=k)
            neighbor_paths = gt_neighbor_paths
            train_path_nums = len(neighbor_paths)
            self.stage = STAGES[2]

        elif level == 'ACTION':
            gt_neighbor_paths, gt_scores = self.GT_action_path_manager.find_similar_paths_by_embedding(test_path, k=k, )
            neighbor_paths = gt_neighbor_paths
            train_path_nums = len(neighbor_paths)
            self.stage = STAGES[3]

        elif level == 'ENTITY':
            gt_neighbor_paths, gt_scores = self.GT_entity_path_manager.find_similar_paths_by_embedding(test_path, k=k, )
            if example_source == 'mix':
                # print("Selecting ENTITY-LEVEL examples from GT and non-GT knowledge bases.")
                non_gt_neighbor_paths, non_gt_scores = self.non_GT_entity_path_manager.find_similar_paths_by_embedding(test_path,
                                                                                                                       k=k,
                                                                                                                       )
                neighbor_paths , _ = self.mix_top_k_neighbors(gt_neighbor_paths, non_gt_neighbor_paths, gt_scores,
                                                          non_gt_scores, k)
                train_path_nums = sum([1 for n_path in neighbor_paths if n_path in gt_neighbor_paths])

            else:
                neighbor_paths = gt_neighbor_paths
                train_path_nums = len(neighbor_paths)
            self.stage = STAGES[4]
        else:
            raise NotImplementedError

        prompt = self.format_detection_prompt(test_path, neighbor_paths, prompt = PROCESS_SEQ_DETECT_PROMPT_WITHOUT_SUMMARIES)

        # print("=== Detecting process sequence: Prompt WITHOUT SUMMARIES===")
        # print(prompt)

        example_seq_ids = set()
        for neighbor_path in neighbor_paths:
            example_seq_ids.update(neighbor_path.sequence_ids)

        call_detail = {"call_type": 'detect', "path_level": test_path.level, "path_identifier": test_path.overall_identifier,
                       "logkey_seq": test_path.find_logkey_sequence_str(), "seq_ids": list(test_path.sequence_ids),
                       "example_pattern_num": len(neighbor_paths),
                       "example_seq_ids": example_seq_ids}

        response = self.llm(prompt=prompt, call_detail=call_detail)
        print(response.output)

        default_json= {
            "prediction": 'Abnormal',
        "reason": 'LLM FAILED'
        }
        response_dict = response.to_json(default_json)
        llm_pred = 1 if response_dict["prediction"] == 'Abnormal' else 0

        print(f"     DETECTION EXAMPLES: {len(neighbor_paths)} example paths, train paths: {train_path_nums}, test paths: {len(neighbor_paths) - train_path_nums}.")
        print(f"     DETECTION RESULT: {response_dict['prediction']}. ")
        print(f"     DETECTION REASON: {response_dict['reason']}. ")

        pred_reason = response_dict["reason"]

        test_path.path_pred = llm_pred
        test_path.path_pred_reason = pred_reason
        test_path.llm_predicted = True

        return llm_pred

    def find_entity_diff_path(self, entity_path: KroneSeq, seq_id):
        diff_paths = []
        cur_diff_nodes = []
        cur_diff_children_paths = []
        if_diff = False
        for i, node in enumerate(entity_path.node_list):
            if i < len(entity_path.node_list) -1:
                next = entity_path.node_list[i+1]
                if if_diff:  # inside a diff segment
                    cur_diff_nodes.append(node)
                    cur_diff_children_paths.append(entity_path.children_paths[i])
                    if next.node_identifier in node.outgoing_neighbors.keys():
                        diff_segment_path = self.GT_entity_path_manager.generate_temp_path(node_list=cur_diff_nodes,
                                                                                          children_paths=cur_diff_children_paths,
                                                                                          seq_id=seq_id, if_GT=False)
                        diff_paths.append(diff_segment_path) # finish and pop out a segment
                        cur_diff_nodes = []
                        cur_diff_children_paths = []
                        if_diff = False
                else:
                    if next.node_identifier not in node.outgoing_neighbors.keys(): # start a diff segment
                        cur_diff_nodes.append(node)
                        cur_diff_children_paths.append(entity_path.children_paths[i])
                        if_diff = True

        return diff_paths


    def mix_top_k_neighbors(self, gt_paths, non_gt_paths, gt_scores, non_gt_scores, k: int) ->Tuple[List[KroneSeq], List[float]]:
        all_paths = gt_paths + non_gt_paths
        all_scores = gt_scores + non_gt_scores

        path_score_pairs = list(zip(all_paths, all_scores))
        sorted_path_score_pairs = sorted(path_score_pairs, key=lambda x: x[1], reverse=True)
        top_k_pairs = sorted_path_score_pairs[:k]
        neighbor_paths = [pair[0] for pair in top_k_pairs]

        return neighbor_paths, all_scores

    def load_test_path_embedding(self, embedding_df: pd.DataFrame):

        self.non_GT_status_path_manager.embedding_df = embedding_df[(embedding_df["path_layer"] == 'STATUS')]
        self.non_GT_action_path_manager.embedding_df = embedding_df[(embedding_df["path_layer"] == 'ACTION')]
        self.non_GT_entity_path_manager.embedding_df = embedding_df[(embedding_df["path_layer"] == 'ENTITY')]
        loaded = len(self.non_GT_status_path_manager.embedding_df) + len(self.non_GT_action_path_manager.embedding_df) + len(self.non_GT_entity_path_manager.embedding_df)
        print(f"Loading {loaded} historical test path embedding")

    def load_test_path_knowledge(self, knowledge_df: pd.DataFrame):

        self.non_GT_status_path_manager.llm_knowledge_df = knowledge_df[(knowledge_df["path_layer"] == 'STATUS')]
        self.non_GT_action_path_manager.llm_knowledge_df = knowledge_df[(knowledge_df["path_layer"] == 'ACTION')]
        self.non_GT_entity_path_manager.llm_knowledge_df = knowledge_df[(knowledge_df["path_layer"] == 'ENTITY')]
        loaded = len(self.non_GT_status_path_manager.llm_knowledge_df) + len(self.non_GT_action_path_manager.llm_knowledge_df) + len(self.non_GT_entity_path_manager.llm_knowledge_df)
        print(f"Loading {loaded} historical test path knowledge")

    def load_test_path_summary(self, test_summary: pd.DataFrame):

        self.non_GT_status_path_manager.summary_df = test_summary[(test_summary["path_layer"] == 'STATUS')]
        self.non_GT_action_path_manager.summary_df = test_summary[(test_summary["path_layer"] == 'ACTION')]
        self.non_GT_entity_path_manager.summary_df = test_summary[(test_summary["path_layer"] == 'ENTITY')]
        loaded = len(self.non_GT_status_path_manager.summary_df) + len(self.non_GT_action_path_manager.summary_df) + len(self.non_GT_entity_path_manager.summary_df)
        print(f"Loading {loaded} historical test path summary")


    def load_or_generate_train_knowledge(self, store_path, knowledge_df: pd.DataFrame = None, ):
        path_knowledge = pd.DataFrame()
        if knowledge_df is not None:
            self.GT_status_path_manager.llm_knowledge_df = knowledge_df[
                (knowledge_df["if_GT"] == True) & (knowledge_df["path_layer"] == 'STATUS')]
            self.GT_status_path_manager.embedding_df = self.GT_status_path_manager.llm_knowledge_df
            self.GT_status_path_manager.summary_df = self.GT_status_path_manager.llm_knowledge_df

            self.GT_action_path_manager.llm_knowledge_df = knowledge_df[
                (knowledge_df["if_GT"] == True) & (knowledge_df["path_layer"] == 'ACTION')]
            self.GT_action_path_manager.embedding_df = self.GT_action_path_manager.llm_knowledge_df
            self.GT_action_path_manager.summary_df = self.GT_action_path_manager.llm_knowledge_df

            self.GT_entity_path_manager.llm_knowledge_df = knowledge_df[
                (knowledge_df["if_GT"] == True) & (knowledge_df["path_layer"] == 'ENTITY')]
            self.GT_entity_path_manager.embedding_df = self.GT_entity_path_manager.llm_knowledge_df
            self.GT_entity_path_manager.summary_df = self.GT_entity_path_manager.llm_knowledge_df

            # Status level
            for path in self.GT_status_path_manager.paths.values():
                self.GT_status_path_manager.load_path_embedding_for_path(path)
                self.GT_status_path_manager.load_path_summary_for_path(path)
                # self.GT_status_path_manager.load_path_prediction_for_path(path)
            for path in self.GT_action_path_manager.paths.values():
                self.GT_action_path_manager.load_path_embedding_for_path(path)
                self.GT_action_path_manager.load_path_summary_for_path(path)
                # self.GT_action_path_manager.load_path_prediction_for_path(path)
            for path in self.GT_entity_path_manager.paths.values():
                self.GT_entity_path_manager.load_path_embedding_for_path(path)
                self.GT_entity_path_manager.load_path_summary_for_path(path)
                # self.GT_entity_path_manager.load_path_prediction_for_path(path)


        # status sequences
        status_generated = self.GT_status_path_manager.batch_generate_embeddings()
        status_generated= self.GT_status_path_manager.summarize_GT_paths(hardcode_kleene_pattern_summary=self.hardcode_kleene_pattern_summary)|status_generated
        status_knowledge = self.GT_status_path_manager.store_training_path_knowledge()
        if status_generated:
            status_knowledge.to_csv(store_path, index=False)
            path_knowledge = pd.concat([path_knowledge, status_knowledge])

        # action sequences
        action_generated= self.GT_action_path_manager.batch_generate_embeddings()
        action_generated= self.GT_action_path_manager.summarize_GT_paths(hardcode_kleene_pattern_summary=self.hardcode_kleene_pattern_summary)|action_generated
        action_knowledge = self.GT_action_path_manager.store_training_path_knowledge()
        if action_generated:
            action_knowledge.to_csv(store_path,mode='a',header=False, index=False)
            path_knowledge = pd.concat([path_knowledge, action_knowledge])

        # entity sequences
        entity_generated =  self.GT_entity_path_manager.batch_generate_embeddings()
        entity_knowledge = self.GT_entity_path_manager.store_training_path_knowledge()
        path_knowledge = pd.concat([path_knowledge, entity_knowledge])
        if entity_generated:
            entity_knowledge.to_csv(store_path,mode='a',header=False, index=False)
            path_knowledge = pd.concat([path_knowledge, entity_knowledge])
        return path_knowledge

    def format_detection_prompt(self, test_path: KroneSeq, neighbor_paths: List[KroneSeq], prompt: str):
        contain_summary = prompt != PROCESS_SEQ_DETECT_PROMPT_WITHOUT_SUMMARIES

        real_k = len(neighbor_paths)
        example_sequence_context = ''
        if len(neighbor_paths) > 0:
            for i, n_path in enumerate(neighbor_paths):
                input_process_seq = n_path.generated_process_sequence_text()
                example_seq_str = f"Normal Example sequence {i} - sequence: "
                example_seq_str += input_process_seq
                example_sequence_context += example_seq_str + "\n"


        # constructing test
        input_process_seq = test_path.generated_process_sequence_text()
        prompt_varibales = {
            "example_num": real_k,
            "example_sequence_context": example_sequence_context,
            "input_process_seq": input_process_seq,
            "DELIMITER": DELIMITER,
        }
        if contain_summary:
            input_process_desc = test_path.path_summary
            prompt_varibales["input_process_desc"] = input_process_desc

        for key, val in prompt_varibales.items():
            prompt = prompt.replace("{" + f"{key}" + "}", f'{val}')

        print(example_sequence_context)
        print(input_process_seq)

        return prompt


    def store_GT_summaries(self):

        GT_status_summaries = self.GT_status_path_manager.store_training_path_knowledge()
        GT_action_summaries = self.GT_action_path_manager.store_training_path_knowledge()
        GT_entity_summaries = self.GT_entity_path_manager.store_training_path_knowledge()
        GT_summaries = pd.concat([GT_status_summaries,GT_action_summaries, GT_entity_summaries])

        return GT_summaries



    def save(self, path):
        pickle.dump(self, open(path, "wb"))
        print(f"Knowledge Base store to {path}!")

    def print_base_info(self):
        print(self.stage)
        print(f"Number of GT status paths: {len(self.GT_status_path_manager.paths.values())}")
        print(f"Number of GT action paths: {len(self.GT_action_path_manager.paths.values())}")
        print(f"Number of GT entity paths: {len(self.GT_entity_path_manager.paths.values())}")
        print(f"Number of non GT status paths: {len(self.non_GT_status_path_manager.paths.values())}")
        print(f"Number of non GT action paths: {len(self.non_GT_action_path_manager.paths.values())}")
        print(f"Number of non GT entity paths: {len(self.non_GT_entity_path_manager.paths.values())}")

# if __name__ == '__main__':
    # dataset = 'HDFS'
    # kb_path = f"../output/{dataset}/knowledgeBase.pkl"
    # kBase = KnowledgeBase()
    # # with open(kb_path, 'rb') as file:
    # #     print("Loading KnowledgeBase...")
    # #     kBase = pickle.load(file)
    # #
    # # kBase.store_path_summary(if_GT=True, layer='STATUS', path = f"../output/{dataset}/GT_status_path_summaries.csv")
    # # kBase.store_path_summary(if_GT=False, layer='STATUS', path = f"../output/{dataset}/non_GT_status_path_summaries.csv")
    #
    # summaries = pd.read_csv(f"../output/{dataset}/GT_status_path_summaries.csv")
    # kBase.load_path_summaries(summaries, if_GT=True, layer='STATUS')
