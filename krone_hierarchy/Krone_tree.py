import ast
import math
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from  executor.time_tracker import TimeTracker

from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_auc_score, confusion_matrix
import time
from tqdm import tqdm

from llm.llm import *
import pickle
from utils import test_metrics
from krone_hierarchy.KnowledgeBase import *
from krone_hierarchy.Node import *
from krone_hierarchy.Krone_seq import *
from krone_hierarchy.PROMPTS import *
DELIMITER = '|'

DEFAULT_ENTITY_TYPES = ["SYSTEM MODULE", "OBJECT"]
class KroneTree(object):

    def __init__(self, time_tracker: TimeTracker, hardcode_kleene_pattern_summary=False, ):
        self.llm = LLM(model="gpt-3.5-turbo")
        self.layers: Dict[str, Dict[str, Node]] = {
            "ENTITY": {},
            "ACTION": {},
            "STATUS": {}
        }
        self.none_count = 0
        self.hardcode_kleene_pattern_summary = hardcode_kleene_pattern_summary
        self.template_id_to_entity: Dict[str, Node] = {}

        self.root_node: Node = None
        self.time_tracker = time_tracker
        self.knowledgeBase: KnowledgeBase = KnowledgeBase(llm = self.llm,time_tracker = self.time_tracker, hardcode_kleene_pattern_summary= hardcode_kleene_pattern_summary)
        self.batch_id = 0
        self.structured_process = None

        self.train_status_nodes = None


    def refine_iter(self, refined_entity_col, refined_action_col, sequences, seq_ids):

        refine_entities_df, updated_actions = self.refine_entities(enable_llm=False, refined_entity_col=refined_entity_col)
        refine_actions_df, updated_actions = self.refine_actions(refine_entities_df, enable_llm=False,
                                                              refined_action_col=refined_action_col,
                                                              refined_entity_col=refined_entity_col)

        refined_processes = { "status": [],
                             refined_entity_col: [], refined_action_col: [],
                             "event_id": [], "log_template": [], "summary": []}

        for status in self.layers['STATUS'].values():
            refined_processes["status"].append(status.node_identifier.split("_")[0])

            refined_entity = refine_entities_df[
                refine_entities_df["entity_node_id"] == status.parent.parent.node_identifier][refined_entity_col].item()
            refined_processes[refined_entity_col].append(refined_entity)

            refined_action = refine_actions_df[
                refine_actions_df["action_node_id"]== status.parent.node_identifier][refined_action_col].item()
            refined_processes[refined_action_col].append(refined_action)

            refined_processes["event_id"].append(list(status.template_ids)[0])
            refined_processes["log_template"].append(status.template)
            refined_processes["summary"].append(status.template_summary)

        refined_processes = pd.DataFrame(refined_processes)
        self.__init__()  # reset krone_hierarchy
        self.construct(refined_processes, refined_entity_col, refined_action_col, sequences, seq_ids)
        self.train_status_nodes = self.inject_sequences(sequences, seq_ids)
        return refined_processes

    def recursive_refine(self, sequences, seq_ids):
        refined_entity_col = "refined_graph_entity"
        refined_action_col = "refined_graph_action"

        print("========== Refining Graph... =========")
        continue_recursion = True
        i = 0
        refined_processes = None
        last_entity_num = 0
        last_action_num = 0
        while continue_recursion:
            print(f"========== Refinement Iter {i} =========")
            refined_processes = self.refine_iter(refined_entity_col, refined_action_col, sequences, seq_ids)
            curr_entity_num = len(refined_processes[refined_entity_col].unique())
            curr_action_num = len(refined_processes[refined_action_col].unique())
            if curr_entity_num == last_entity_num and curr_action_num == last_action_num:
                continue_recursion = False
            last_entity_num = curr_entity_num
            last_action_num = curr_action_num
            i+=1
        print(f"========== Refinement Finished, recursion {i} =========")
        return refined_processes


    def construct(self, structured_process: pd.DataFrame, entity_col: str, action_col:str ):
        prints ='========== Buidling Graph... ========== \n'
        print(prints)
        self.structured_process = structured_process
        self.root_node = Node(-1, "ROOT", "ROOT")
        self.root_node.template_ids = structured_process["event_id"].tolist()

        non_none_entity_process = structured_process[~(
                (structured_process[entity_col]=='none') | (structured_process[entity_col]==
                'None'))]
        none_entity_process = structured_process[
            (structured_process[entity_col]=='none') | (structured_process[entity_col] =='None')]

        entities = non_none_entity_process[entity_col].unique()
        for e_name in entities:  # build the krone_hierarchy by trees
            entity_rows = self.structured_process[entity_col] == e_name
            entity_process = self.structured_process[entity_rows]
            entity_template_ids = set(entity_process["event_id"].astype(str).unique())
            entity_node_id = len(self.layers["ENTITY"])
            entity_node = self.create_node(name=e_name, node_id=entity_node_id, node_type="ENTITY",
                                           t_ids=entity_template_ids,
                                           )
            self.root_node.add_child(entity_node)

            non_none_action_process = entity_process[
                ~ ((entity_process[action_col] =='none') | entity_process[action_col]=='None')]
            none_action_process = entity_process[
                (entity_process[action_col] =='none') | entity_process[action_col]=='None']
            v_actions = non_none_action_process[action_col].unique()
            for action in v_actions:
                action_rows = entity_process[action_col] == action
                action_process = entity_process[action_rows]
                action_node_id = len(self.layers["ACTION"])
                action_template_ids = set(action_process["event_id"].astype(str).unique())
                action_node = self.create_node(name=action, node_id=action_node_id, node_type="ACTION",
                                               t_ids=action_template_ids)

                entity_node.add_child(action_node)
                for row_id, row in action_process.iterrows():
                    status_node_id = len(self.layers["STATUS"])
                    is_anomaly = row['is_anomaly'] if 'is_anomaly' in row else False
                    is_anomaly_reason = row['is_anomaly_reason'] if is_anomaly else ''
                    status_node = self.create_node(name=row["status"], node_id=status_node_id, node_type="STATUS",
                                                   t_ids={str(row["event_id"])}, template=row["log_template"],
                                                   template_summary=row["summary"], is_anomaly = is_anomaly,is_anomaly_reason = is_anomaly_reason,
                                                   row_id = row_id)
                    action_node.add_child(status_node)

            # create individual action and status node for each none action node
            for row_id, none_action_row in none_action_process.iterrows():
                self.create_nodes_for_none_entity_or_none_action_row(none_action_row, row_id, 'ACTION', entity_node, entity_col,
                                                                     action_col)

        for row_id, row in none_entity_process.iterrows():
            self.create_nodes_for_none_entity_or_none_action_row(row, row_id,'ENTITY', self.root_node, entity_col, action_col)

    def output_graph(self, output_path):

        entity_ids: List[str] = []  # the entity path
        action_ids: List[str] = []  # the action path
        status_ids: List[str] = []  # the status path

        for t_id in self.structured_process["event_id"].tolist():
            entity = self.template_id_to_entity[str(t_id)]
            nodes = [entity]
            nodes = entity.find_nodes_for_t_id(t_id, nodes)
            assert len(nodes) == 3
            action = nodes[1]
            status = nodes[-1]
            entity_ids.append(entity.node_identifier)
            action_ids.append(action.node_identifier)
            status_ids.append(status.node_identifier)

        self.structured_process["entity_node_id"] = entity_ids
        self.structured_process["action_node_id"] = action_ids
        self.structured_process["status_node_id"] = status_ids
        self.structured_process.to_csv(output_path, index=False)




    def node_detection(self, structured_process:pd.DataFrame):
        modified = False
        if 'is_anomaly' not in structured_process.columns:
            print("========== Detecting for log templates... ==========")
            tuples = []
            for status_node in self.layers['STATUS'].values():
                tuple = (list(status_node.template_ids)[0] , status_node)
                tuples.append(tuple)
            tuples.sort(key=lambda x: x[0])

            row_id_to_prediction = {"row_id":[], "is_anomaly":[], "is_anomaly_reason":[]}
            for (t_id, status_node) in tuples:
                prompt_variables = {"input_process": status_node.template}
                prompt = self.format_prompt(PROCESS_DETECT_PROMPT, prompt_variables)
                # print(prompt)
                default_json = {"prediction": "Normal",
                                "reason": "LLM FAILED"}
                response = self.llm(prompt, default_json=default_json)
                print(f"template id: {t_id}, template {status_node.template}")
                print(response.output)
                result_json = response.to_json(default_json=default_json)
                status_node.is_anomaly = result_json['prediction'] =='Abnormal'
                status_node.is_anomaly_reason = result_json['reason']
                row_id_to_prediction["row_id"].append(status_node.row_id)
                row_id_to_prediction["is_anomaly"].append(status_node.is_anomaly)
                row_id_to_prediction["is_anomaly_reason"].append(result_json['reason'])

            row_id_to_prediction = pd.DataFrame(row_id_to_prediction)
            row_id_to_prediction = row_id_to_prediction.sort_values(by = ['row_id'])
            structured_process["is_anomaly"] = row_id_to_prediction["is_anomaly"]
            structured_process["is_anomaly_reason"] = row_id_to_prediction["is_anomaly_reason"]
            modified = True
        self.print_graph()

        return structured_process, modified


    def create_nodes_for_none_entity_or_none_action_row(self, row, row_id,  none_level, parent_node, entity_col, action_col):
        if none_level =='ENTITY': # a row of none entity
            n_entity_node_id = len(self.layers["ENTITY"])
            n_entity_template_ids = {str(row["event_id"])}
            # just in case the name is "none_0" instead of "none"
            entity_name = row[entity_col] if '_' not in row[entity_col] else \
                row[entity_col].split("_")[0]
            n_entity_node = self.create_node(name=entity_name, node_id=n_entity_node_id, node_type="ENTITY",
                                             t_ids=n_entity_template_ids)
            self.create_nodes_for_none_entity_or_none_action_row(row, row_id, 'ACTION', n_entity_node, entity_col, action_col)

        elif none_level =='ACTION':
            n_action_node_id = len(self.layers["ACTION"])
            n_action_template_ids = {str(row["event_id"])}
            # just in case the name is "none_0" instead of "none"
            action_name = row[action_col] if '_' not in row[action_col] else \
            row[action_col].split("_")[0]
            n_action_node = self.create_node(name=action_name, node_id=n_action_node_id, node_type="ACTION",
                                             t_ids=n_action_template_ids)

            parent_node.add_child(n_action_node)  # add none action node to entity
            # create status node
            status_node_id = len(self.layers["STATUS"])
            status_name = row['status'] if '_' not in row['status'] else \
            row['status'].split("_")[0]
            is_anomaly = row['is_anomaly'] if 'is_anomaly' in row else False
            is_anomaly_reason = row['is_anomaly_reason'] if is_anomaly else ''
            status_node = self.create_node(name=status_name, node_id=status_node_id, node_type="STATUS",
                                           t_ids=n_action_template_ids, template = row["log_template"], template_summary =row["summary"],
                                           row_id=row_id, is_anomaly = is_anomaly, is_anomaly_reason=is_anomaly_reason)
            n_action_node.add_child(status_node)
        else:
            raise ValueError(f"none_level {none_level} is not valid")

    def create_node(self, name: str, node_id: int, node_type,  t_ids: Set[str] = None,  template = None,
                    template_summary = None, is_anomaly = False,is_anomaly_reason='', row_id = -1):
        # if belongs to the same tree
        name = name.split("_")[0]
        node_identifier = name +"_"+str(node_id)
        node = Node(node_id, node_identifier, node_type, t_ids,  template, template_summary, is_anomaly=is_anomaly,is_anomaly_reason=is_anomaly_reason, row_id=row_id)
        self.layers[node_type][node_identifier] = node
        if node_type =='ENTITY':
            for t_id in t_ids:
                self.template_id_to_entity[t_id] = node
        return node

    def inject_sequences(self, sequences: list[list[str]], seq_ids: List[int], knowledge_df = None):
        print(f"========== Injecting Training Sequences... =========")
        total_entity_nodes_list: List[List[Node]] = []
        total_action_nodes_list: List[List[Node]] = []
        total_status_nodes_list: List[List[Node]]  = []

        normal_status_nodes: Set[Node] = set()
        start = time.time()
        for sequence in tqdm(sequences):  # retrieve entity and status paths
            entity_nodes = [] # the entity path
            action_nodes = [] # the status path
            status_nodes = [] # the status path
            for t_id in sequence:
                if t_id ==1:
                    print("??")
                root = self.template_id_to_entity[str(t_id)]
                nodes = [root]
                nodes = root.find_nodes_for_t_id(t_id, nodes)
                nodes[-1].is_anomaly = False  # status node is not anomaly if appears in training
                assert len(nodes) == 3
                entity_nodes.append(root)
                action_nodes.append(nodes[1])
                status_nodes.append(nodes[-1])
                normal_status_nodes.add(nodes[-1])
            total_entity_nodes_list.append(entity_nodes)
            total_action_nodes_list.append(action_nodes)
            total_status_nodes_list.append(status_nodes)

        for entity_nodes, action_nodes, status_nodes, seq_id in tqdm(zip(total_entity_nodes_list, total_action_nodes_list, total_status_nodes_list, seq_ids)):
            # for a sequence
            assert len(entity_nodes) == len(action_nodes) == len(status_nodes)
            _,_,_ = self.knowledgeBase.create_paths_for_sequence(entity_nodes, action_nodes, status_nodes, seq_id, if_GT=True)
        end = time.time()
        print(f"========== Finished Injecting {len(sequences)} Sequences, Duration {end - start} =========")
        print(f"Status path: {len(self.knowledgeBase.GT_status_path_manager.status_identifier_to_paths)}")
        print(f"Action path: {len(self.knowledgeBase.GT_action_path_manager.action_identifier_to_paths)}")
        print(f"Entity path: {len(self.knowledgeBase.GT_entity_path_manager.entity_identifier_to_paths)}")
        if knowledge_df is not None:
            self.knowledgeBase.load_test_path_knowledge(knowledge_df, if_GT=True)

        self.print_graph()
        self.print_tree()
        return normal_status_nodes



    def refine_entities(self, enable_llm, refined_entity_col):
        # if an action node only has one outgoing neighbor, examine if they are the same
        refine_entities = {"entity_node_id": [],
                           refined_entity_col: [], "refined_entity_node_source": [],
                           }
        print("========== Refining Entities... =========")
        for entity_node in self.layers["ENTITY"].values():
            if entity_node.node_identifier not in refine_entities["entity_node_id"]:
                strong_chain = entity_node.strong_chain()
                possible_merge = len(strong_chain) > 1
                if possible_merge:
                    print(f"Current entity: {entity_node.node_identifier}")
                    print(f"Entity {len(strong_chain)} Strong chain: {[chain_node.node_identifier for chain_node in strong_chain]}")
                    if enable_llm:
                        process_strs = []
                        i = 0
                        for chain_node in strong_chain:
                            for action in chain_node.children.values():
                                for status in action.children.values():
                                    process_str = f"Process {i}: " + status.template_summary + DELIMITER + status.parent.parent.get_semantic_node_name()
                                    process_strs.append(process_str)
                                    i += 1

                        final_process = "\n".join(process_strs)
                        prompt_variables = {"process_num": i,
                                            "process_list": final_process,
                                            "entity_types": DEFAULT_ENTITY_TYPES}
                        prompt = self.format_prompt(ENTITY_MERGE_PROMPT_SOFT, prompt_variables)
                        # print(prompt)
                        default_json = {"decision": "False",
                                        "entity": "none",
                                        "entity_source": "none"}
                        response = self.llm(prompt, default_json=default_json)
                        # print(response.output)
                        result_json = response.to_json(default_json=default_json)
                        common_entity = result_json["entity"]
                        source = result_json["entity_source"]
                        for chain_node in strong_chain:
                            if result_json["decision"] == 'True':  # chain can be merged
                                refine_entities["entity_node"].append(chain_node.node_identifier)
                                refine_entities[refined_entity_col].append(common_entity)
                                refine_entities["refined_entity_node_source"].append(source)
                            else:  # chain cannot be merged
                                refine_entities["entity_node"].append(chain_node.node_identifier)
                                refine_entities[refined_entity_col].append(chain_node.node_identifier)
                                refine_entities["refined_entity_node_source"].append('none')
                    else:
                        common_entity = ''.join([chain_node.get_semantic_node_name().capitalize() for chain_node in strong_chain])
                        source ='MERGE'
                        for chain_node in strong_chain:
                            if chain_node.node_identifier not in refine_entities["entity_node_id"]:
                                refine_entities["entity_node_id"].append(chain_node.node_identifier)
                                refine_entities[refined_entity_col].append(common_entity)
                                refine_entities["refined_entity_node_source"].append(source)


                else: # current entity has no chain
                    refine_entities["entity_node_id"].append(entity_node.node_identifier)
                    refine_entities[refined_entity_col].append(entity_node.node_identifier.split("_")[0])
                    refine_entities["refined_entity_node_source"].append('none')

            else:
                continue # visited

        print(f"Entity Refinement Finished! {len(set(refine_entities['entity_node_id']))} -> {len(set(refine_entities[refined_entity_col]))} entities!")
        updated = len(set(refine_entities['entity_node_id'])) != len(set(refine_entities[refined_entity_col]))
        refine_entities = pd.DataFrame(refine_entities)
        return refine_entities, updated

    def refine_actions(self, refine_entities, enable_llm, refined_action_col, refined_entity_col):
        print("========== Refining Actions... =========")
        refine_actions = {"action_node_id": [], refined_action_col: [], "refined_action_node_source": [], }

        for action_node in self.layers["ACTION"].values():
            if action_node.node_identifier not in refine_actions["action_node_id"]:
                strong_chain = action_node.strong_chain()
                possible_merge = len(strong_chain) > 1
                if possible_merge:
                    print(f"Action Strong chain: {[chain_node.node_identifier for chain_node in strong_chain]}")
                    if enable_llm:
                        all_statues = []

                        for chain_node in strong_chain:
                            all_statues.extend(list(chain_node.children.values()))

                        process_strs = []
                        cur_refined_entity = refine_entities[refined_entity_col][
                            refine_entities["entity_node"].index(action_node.parent.node_identifier)]

                        for i, status in enumerate(all_statues):
                            process_str = f"Process {i}: " + status.template_summary + DELIMITER + cur_refined_entity + DELIMITER + status.parent.get_semantic_node_name()
                            process_strs.append(process_str)
                        final_process = "\n".join(process_strs)
                        prompt_variables = {"process_num": len(all_statues),
                                            "process_list": final_process}
                        prompt = self.format_prompt(ACTION_MERGE_PROMPT_SOFT, prompt_variables)
                        # print(prompt)
                        response = self.llm(prompt)
                        # print(response.output)
                        default_json = {"decision": "False",
                                        "action": "none",
                                        "action_source": "none"}
                        result_json = response.to_json(default_json=default_json)


                        common_action = result_json["action"]
                        source = result_json["action_source"]
                        for chain_node in strong_chain:
                            if result_json["decision"] == 'True':  # chain can be merged
                                refine_actions["action_node"].append(chain_node.node_identifier)
                                refine_actions[refined_action_col].append(common_action)
                                refine_actions["refined_action_node_source"].append(source)
                            else:  # chain cannot be merged
                                refine_actions["action_node"].append(chain_node.node_identifier)
                                refine_actions[refined_action_col].append(chain_node.node_identifier)
                                refine_actions["refined_action_node_source"].append('none')
                    else:
                        common_action = ''.join([chain_node.get_semantic_node_name().capitalize() for chain_node in strong_chain])
                        source ='MERGE'
                        for chain_node in strong_chain:
                            if chain_node.node_identifier not in refine_actions["action_node_id"]:
                                refine_actions["action_node_id"].append(chain_node.node_identifier)
                                refine_actions[refined_action_col].append(common_action)
                                refine_actions["refined_action_node_source"].append(source)

                else: # current action has no chain
                    refine_actions["action_node_id"].append(action_node.node_identifier)
                    refine_actions[refined_action_col].append(action_node.node_identifier.split("_")[0])
                    refine_actions["refined_action_node_source"].append('none')
            else:
                continue  # if visited, do nothing

        print(f"Action Refinement Finished! {len(set(refine_actions['action_node_id']))} -> {len(set(refine_actions[refined_action_col]))} actions!")
        updated = len(set(refine_actions['action_node_id'])) != len(set(refine_actions[refined_action_col]))
        refine_actions = pd.DataFrame(refine_actions)
        return refine_actions, updated

    def format_prompt(self, prompt, prompt_variables):
        for key, val in prompt_variables.items():
            prompt = prompt.replace("{" + f"{key}" + "}", f'{val}')
        return prompt


    def _breakdown_sequence_into_paths(self, sequence: list[str], test_seq_id: int):

        entity_nodes: List[Node] = []  # the entity path
        action_nodes: List[Node] = []  # the action path
        status_nodes: List[Node] = []  # the status path


        for t_id in sequence:
            entity = self.template_id_to_entity[str(t_id)]
            nodes = [entity]
            nodes = entity.find_nodes_for_t_id(t_id, nodes)
            assert len(nodes) == 3
            action = nodes[1]
            status = nodes[-1]
            entity_nodes.append(entity)
            action_nodes.append(action)
            status_nodes.append(status)

        entity_path, action_paths, status_paths, = self.knowledgeBase.create_paths_for_sequence( entity_nodes, action_nodes,
                                                            status_nodes, test_seq_id, if_GT=False)
        return (entity_path, action_paths, status_paths, entity_nodes, action_nodes, status_nodes)




    def detect(self, test, entity_level = True, action_level=True, status_level = True,
               detect_mode_entity ='local', detect_mode_action ='local', detect_mode_status ='local', k_neighbors = 3, example_source ='GT',
               edge_consecutive_sensitive= False, automaton_adjustment= False, dummy_detect= False, dummy_summarize = False, lazy_detect = True):

        test_sequences = test["EventSequence"].map(lambda s: [str(e) for e in ast.literal_eval(s)]).tolist()
        test_seq_ids = test["seq_id"].tolist()
        labels = test["Label"].tolist()

        test_results = {
            "seq_id": [],
            "seq_label": [],
            "detectors": [],  # local or llm
                   "entity_pred": [],
                   "action_pred":[],
                   "status_pred":[],
        "final_pred":[]}

        llm_call_stats = {"seq_id":[], "seq_label":[], "status_calls":[], "action_calls":[], "entity_calls":[], "total_calls":[]}
        start = time.time()
        i = 0
        # all time durations
        total_path_breakdown_duration = 0
        total_status_detection_duration = 0
        total_action_detection_duration = 0
        total_entity_detection_duration = 0
        krone_decompose_res_viz = {"seq_id": [], "seq": [], "entity_nodes_for_logkeys":[],  "action_nodes_for_logkeys":[], "status_nodes_for_logkeys":[]}
        krone_detection_res_viz = {"seq_id": [], "seq": [], "anomaly_seg":[], "anomaly_level":[], "anomaly_reason":[]}
        self.time_tracker.update_sequence_num(len(test_sequences))
        for test_seq, test_seq_id, test_label in zip(test_sequences, test_seq_ids, labels):
            last_time = time.time()

            (entity_path, action_paths, status_paths,  entity_nodes, action_nodes, status_nodes) =\
                self._breakdown_sequence_into_paths(test_seq, test_seq_id)
            self.time_tracker.update_sequence_breakdown(time.time()-last_time)

            assert len(entity_nodes) == len(action_nodes)
            assert len(status_nodes) == len(action_nodes)
            assert len(status_nodes) == len(test_seq)

            krone_decompose_res_viz["seq_id"].append(test_seq_id)
            krone_decompose_res_viz["seq"].append(test_seq)
            krone_decompose_res_viz["entity_nodes_for_logkeys"].append("["+",".join([e.node_identifier for e in entity_nodes])+"]")
            krone_decompose_res_viz["action_nodes_for_logkeys"].append("["+",".join([a.node_identifier for a in action_nodes])+"]")
            krone_decompose_res_viz["status_nodes_for_logkeys"].append("["+",".join([s.node_identifier for s in status_nodes])+"]")

            path_breakdown_duration = (time.time() - last_time)
            total_path_breakdown_duration +=path_breakdown_duration

            # test in three levels
            seq_result = {"local": {"entity_pred": 0, "action_pred": 0, "status_pred": 0},
                      "llm": {"entity_pred": 0, "action_pred": 0, "status_pred": 0}}

            status_llm_calls = 0
            action_llm_calls = 0
            entity_llm_calls = 0

            curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            print(f"******** {curr_time} Detecting for {'Normal' if test_label==0 else 'Abnormal'} seq {test_seq_id}, {i}/{len(test_sequences)} ********")
            if status_level:
                local_status_pred = 0
                llm_status_pred = 0
                curr_time = time.time()
                for status_path in status_paths:

                    mini_local_status_pred, mini_llm_status_pred, mini_llm_call = self.knowledgeBase.general_sequence_detect_v2(test_path = status_path,seq_id = test_seq_id,
                                                                                                                             k = k_neighbors,
                                                                                                                                # automaton_adjustment = automaton_adjustment,
                                                                                                                             example_source= example_source,
                                                                                                                             detect_mode=detect_mode_status,
                                                                                                                            dummy_detect = dummy_detect,
                                                                                                                            dummy_summarize = dummy_summarize
                                                                                                                             )

                    if mini_llm_status_pred ==1:
                        krone_detection_res_viz["seq_id"].append(test_seq_id)
                        krone_detection_res_viz["seq"].append(test_seq)
                        krone_detection_res_viz["anomaly_seg"].append("[" +", ".join(status_path.get_logkeys()) + "]")
                        krone_detection_res_viz["anomaly_level"].append("status")
                        krone_detection_res_viz["anomaly_reason"].append(status_path.path_pred_reason)

                    status_llm_calls += mini_llm_call
                    local_status_pred = local_status_pred | mini_local_status_pred
                    llm_status_pred = llm_status_pred | mini_llm_status_pred
                    if  lazy_detect and llm_status_pred == 1:
                        break
                status_detection_duration = (time.time() - curr_time) # status detection duration
                total_status_detection_duration += status_detection_duration

                seq_result["local"]["status_pred"] = local_status_pred
                seq_result["llm"]["status_pred"] = llm_status_pred

            if lazy_detect:
                if_go_up = action_level and seq_result["llm"]["status_pred"]== 0
            else:
                if_go_up = True

            if if_go_up:

                local_action_pred = 0
                llm_action_pred = 0
                curr_time = time.time()
                for action_path in action_paths:

                    mini_local_action_pred, mini_llm_action_pred, mini_llm_call = self.knowledgeBase.general_sequence_detect_v2(test_path = action_path,seq_id = test_seq_id,
                                                                                                                             k = k_neighbors,
                                                                                                                             detect_mode=detect_mode_action,
                                                                                                                                automaton_adjustment=automaton_adjustment,
                                                                                                                             example_source= example_source,
                                                                                                                                dummy_detect=dummy_detect,
                                                                                                                                dummy_summarize=dummy_summarize
                                                                                                                             )
                    if mini_llm_action_pred ==1:
                        krone_detection_res_viz["seq_id"].append(test_seq_id)
                        krone_detection_res_viz["seq"].append(test_seq)
                        krone_detection_res_viz["anomaly_seg"].append("[" +", ".join(action_path.get_logkeys()) + "]")
                        krone_detection_res_viz["anomaly_level"].append("action")
                        krone_detection_res_viz["anomaly_reason"].append(action_path.path_pred_reason)

                    action_llm_calls += mini_llm_call
                    local_action_pred = local_action_pred | mini_local_action_pred
                    llm_action_pred = llm_action_pred | mini_llm_action_pred
                    if lazy_detect and llm_action_pred == 1:
                        break

                action_detection_duration= (time.time() - curr_time)
                total_action_detection_duration += action_detection_duration

                seq_result["local"]["action_pred"] = local_action_pred
                seq_result["llm"]["action_pred"] = llm_action_pred

            if lazy_detect:
                if_go_up = if_go_up and seq_result["llm"]["action_pred"] == 0 and entity_level
            else:
                if_go_up = True

            if if_go_up:
                curr_time = time.time()
                local_entity_pred, llm_entity_pred, mini_llm_call = self.knowledgeBase.general_sequence_detect_v2(test_path = entity_path, seq_id = test_seq_id,
                                                                                                               k = k_neighbors,
                                                                                                               detect_mode=detect_mode_entity,
                                                                                                               example_source= example_source,
                                                                                                               automaton_adjustment = automaton_adjustment,
                                                                                                               edge_consecutive_sensitive=edge_consecutive_sensitive,
                                                                                                                  dummy_detect=dummy_detect,
                                                                                                                  dummy_summarize=dummy_summarize
                                                                                                               )
                if llm_entity_pred == 1:
                    krone_detection_res_viz["seq_id"].append(test_seq_id)
                    krone_detection_res_viz["seq"].append(test_seq)
                    krone_detection_res_viz["anomaly_seg"].append("[" +", ".join(entity_path.get_logkeys()) + "]")
                    krone_detection_res_viz["anomaly_level"].append("entity")
                    krone_detection_res_viz["anomaly_reason"].append(entity_path.path_pred_reason)

                entity_detection_duration = (time.time() - curr_time)
                total_entity_detection_duration += entity_detection_duration
                entity_llm_calls = mini_llm_call
                seq_result["local"]["entity_pred"] = local_entity_pred
                seq_result["llm"]["entity_pred"] = llm_entity_pred

            llm_call_stats["seq_id"].append(test_seq_id)
            llm_call_stats["seq_label"].append(test_label)
            llm_call_stats["status_calls"].append(status_llm_calls)
            llm_call_stats["action_calls"].append(action_llm_calls)
            llm_call_stats["entity_calls"].append(entity_llm_calls)
            llm_call_stats["total_calls"].append(status_llm_calls+action_llm_calls+entity_llm_calls)

            local_final_pred =  seq_result["local"]["entity_pred"] | seq_result["local"]["action_pred"] | seq_result["local"]["status_pred"]
            local_detection_result = 'PASS' if local_final_pred == test_label else 'FAILED'
            if local_detection_result == 'PASS':
                print(f"******** Local Detection result: \033[1m\033[32m{local_detection_result}\033[0m for GT {'Abnormal' if test_label else 'Normal'} seq {test_seq_id} ********")
            else:
                case ='False Negative' if test_label else 'False Positive'
                print(f"******** Local Detection result: \033[1m\033[31m{local_detection_result}\033[0m for GT {'Abnormal' if test_label else 'Normal'} seq {test_seq_id}, \033[1m\033[38;2;255;140;0m{case}\033[0m ********")

            final_pred =  seq_result["llm"]["entity_pred"] | seq_result["llm"]["action_pred"] | seq_result["llm"]["status_pred"]
            detection_result = 'PASS' if final_pred == test_label else 'FAILED'
            if detection_result == 'PASS':
                print(f"******** LLM Detection result: \033[1m\033[32m{detection_result}\033[0m for GT {'Abnormal' if test_label else 'Normal'} seq {test_seq_id} ********")
            else:
                case ='False Negative' if test_label else 'False Positive'
                print(f"******** LLM Detection result: \033[1m\033[31m{detection_result}\033[0m for GT {'Abnormal' if test_label else 'Normal'} seq {test_seq_id}, \033[1m\033[38;2;255;140;0m{case}\033[0m ********")

            for detector in ["local", "llm"]:
                test_results["seq_id"].append(test_seq_id)
                test_results["seq_label"].append(test_label)
                test_results["detectors"].append(detector)
                test_results["entity_pred"].append(seq_result[detector]["entity_pred"])
                test_results["action_pred"].append(seq_result[detector]["action_pred"])
                test_results["status_pred"].append(seq_result[detector]["status_pred"])
                final = seq_result[detector]["status_pred"] | seq_result[detector]["action_pred"] | seq_result[detector]["entity_pred"]
                test_results["final_pred"].append(final)
            i+=1

        total_duration = total_path_breakdown_duration + total_status_detection_duration + total_action_detection_duration + total_entity_detection_duration
        print(
            f"total duration: {total_duration}, sequence breakdown duration: {total_path_breakdown_duration}, "
            f"status duration: {total_status_detection_duration}, action duration: {total_action_detection_duration}, entity duration: {total_entity_detection_duration}")

        test_results = pd.DataFrame(test_results)
        local_precision, local_recall, local_f1, llm_precision, llm_recall, llm_f1 = self.detect_metrics(test_results,  entity_level, action_level, status_level)
        print(f"LLM requests: {self.knowledgeBase.dummy_llm_detection_calls}")
        print(f"abnormal sequence detected by abnormal nodes: {len(self.knowledgeBase.seq_detected_by_nodes)}")


        return test_results, llm_call_stats, local_precision, local_recall, local_f1, llm_precision, llm_recall, llm_f1, krone_detection_res_viz, krone_decompose_res_viz
    #

    def detect_metrics(self, results: pd.DataFrame,  entity_level = True, action_level=True, status_level = True):
        detectors = ["local", "llm"]
        local_precision, local_recall, local_f1 = 0,0,0
        llm_precision, llm_recall, llm_f1 = 0,0,0
        for detector in detectors:
            total_predictions = np.zeros(int(len(results)/2)).astype(int)
            detector_results = results[results["detectors"]==detector]
            if entity_level:
                entity_predictions = results[results["detectors"]==detector]["entity_pred"]
                entity_test_metrics = test_metrics(entity_predictions, detector_results["seq_label"])
                print(
                    f"Entity ({detector}) Test: F-1: {entity_test_metrics['f1']}, Precision: {entity_test_metrics['p']}, Recall {entity_test_metrics['r']}, TP: {entity_test_metrics['tp']}, FP: {entity_test_metrics['fp']}, TN: {entity_test_metrics['tn']}, FN: {entity_test_metrics['fn']}")
                total_predictions = total_predictions | entity_predictions

            if action_level:
                action_predictions = results[results["detectors"]==detector]["action_pred"]
                action_test_metrics = test_metrics(action_predictions, detector_results["seq_label"])
                print(
                    f"Action ({detector}) Test: F-1: {action_test_metrics['f1']}, Precision: {action_test_metrics['p']}, Recall {action_test_metrics['r']}, TP: {action_test_metrics['tp']}, FP: {action_test_metrics['fp']}, TN: {action_test_metrics['tn']}, FN: {action_test_metrics['fn']}")
                total_predictions = total_predictions | action_predictions
            if status_level:
                status_predictions = results[results["detectors"]==detector]["status_pred"]
                status_test_metrics = test_metrics(status_predictions, detector_results["seq_label"])
                print(
                    f"Status ({detector}) Test: F-1: {status_test_metrics['f1']}, Precision: {status_test_metrics['p']}, Recall {status_test_metrics['r']}, TP: {status_test_metrics['tp']}, FP: {status_test_metrics['fp']}, TN: {status_test_metrics['tn']}, FN: {status_test_metrics['fn']}")
                total_predictions = total_predictions | status_predictions

            total_test_metrics = test_metrics(total_predictions, detector_results["seq_label"])
            print(
                f"Total ({detector}) Test: F-1: {total_test_metrics['f1']}, Precision: {total_test_metrics['p']}, Recall {total_test_metrics['r']}, TP: {total_test_metrics['tp']}, FP: {total_test_metrics['fp']}, TN: {total_test_metrics['tn']}, FN: {total_test_metrics['fn']}")

            if detector =='local':
                local_precision, local_recall, local_f1 = total_test_metrics['p'], total_test_metrics['r'], total_test_metrics['f1']
            elif detector =='llm':
                llm_precision, llm_recall, llm_f1 = total_test_metrics['p'], total_test_metrics['r'], \
                total_test_metrics['f1']
            else:
                raise  NotImplementedError

        return local_precision, local_recall, local_f1, llm_precision, llm_recall, llm_f1


    def print_graph(self):

        # print_tree(self.root_node)
        # print(f"LLM detection calls: {self.knowledgeBase.llm_calls}")
        # entities = set(krone_hierarchy.layers["ENTITY"].values())
        # for entity_node in entities:
        #     print_tree(entity_node)

        anomaly_nodes = []
        for node in self.layers["STATUS"].values():
            if node.masked:
                print(f"Status node {node.node_identifier} is masked!")
            if node.is_anomaly:
                print(f"Status node {node.node_identifier}, {list(node.template_ids)[0]} is anomaly, reason: {node.is_anomaly_reason}!")
                anomaly_nodes.append(list(node.template_ids)[0])
        print(f"Entity layer: {len(self.layers['ENTITY'].keys())}")
        print(f"Action layer: {len(self.layers['ACTION'].keys())}")
        print(f"Status layer: {len(self.layers['STATUS'].keys())}, anomaly nodes: {len(anomaly_nodes)}")
        anomaly_nodes.sort()
        print(f"All abnormal nodes: {anomaly_nodes}")

    def print_tree(self, node:Node=None):
        if node is None:
            node = self.root_node  # Set default node to root_node

        node_str = '"'+node.node_identifier+'"'+f"layer: {node.node_type}, children: {len(node.children)}"+": ["
        children_str = ','.join(['"'+child+'"' for child, _ in node.children.items()])
        node_str += children_str
        node_str += "],"
        print(node_str)
        for child in node.children.values():
            self.print_tree(child)


    def save(self, path):
        pickle.dump(self, open(path, "wb"))
        print(f"Graph store to {path}!")


# todo: separate llm and retrieve knowledge so that a local detector can be applied without accessing llm

