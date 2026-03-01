import datetime
import os.path

from krone_hierarchy.Krone_tree import KroneTree
import ast
import pandas as pd
import numpy as np
import math
from utils import test_metrics
from tqdm import tqdm
from  executor.time_tracker import TimeTracker

from datetime import datetime
class Tee(object):
    def __init__(self, file, stream):
        self.file = file  # File stream
        self.stream = stream  # Console stream (e.g., sys.stdout)

    def write(self, data):
        # Write data to both file and console
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        # Ensure data is written out (needed for compatibility)
        self.file.flush()
        self.stream.flush()

def create_paths(output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

def generate_random_string(number_file_path, num_words=3, ):
    from random_word import RandomWords
    import random
    number = 0
    if not os.path.exists(number_file_path):
        with open(number_file_path, 'w') as f:
            f.write("0")
    else:
        with open(number_file_path, 'r') as f:
            number = int(f.readline().strip()) +1
        with open(number_file_path, 'w') as f:
            f.write(str(number))
    r = RandomWords()
    words = [r.get_random_word() for _ in range(num_words)]
    number = str(number)
    project_name = number+'_'+'_'.join(words)

    return project_name

class Executor(object):
    def __init__(self):
        self.dataset =''
        self.entity_level = True
        self.action_level =True
        self.status_level =True
        self.detect_mode ='local'
        self.k_neighbors= 5
        self.graph = None
        self.output_path = ''
        self.log_file = None
        self.train_percent = 0
        self.test_1_percent = 0
        self.project_name = ''

        self.train, self.test_1, self.test_2 = None, None, None

        self.curr_test_embedding = None
        self.curr_train_knowledge = None
        self.curr_test_knowledge = None
        self.test_time_tracker = TimeTracker()

    def load_configs(self, configs):
        import pprint

        self.dataset = configs['dataset']
        self.entity_level = configs['entity_level']
        self.action_level = configs['action_level']
        self.status_level = configs['status_level']
        self.lazy_detect = configs['lazy_detect']
        self.hardcode_kleene_pattern_summary = configs["hardcode_kleene_pattern_summary"] if "hardcode_kleene_pattern_summary" in configs else False
        self.train_percent = configs['train_percent']
        self.test_1_percent = configs['test_1_percent']
        self.detect_mode = configs['detect_mode']
        self.k_neighbors = configs['k_neighbors']
        self.load_history_test_knowledge = configs['load_history_test_knowledge']
        self.load_history_test_summary = configs['load_history_test_summary']
        self.load_history_test_embedding = configs['load_history_test_embedding']
        self.store_test_knowledge_as_history = configs['store_test_knowledge_as_history']
        self.automaton_adjustment = configs['automaton_adjustment']
        self.edge_consecutive_sensitive = configs['edge_consecutive_sensitive']

        self.dummy_summarize = configs['dummy_summarize'] if 'dummy_summarize' in configs else False
        self.dummy_detect = configs['dummy_detect'] if 'dummy_detect' in configs else False


        self.hist_test_embedding = None
        self.hist_test_summary = None
        self.hist_test_knowledge = None

        import datetime
        print(datetime.datetime.now())
        pprint.pprint(configs, sort_dicts=False)
        if (not self.lazy_detect) and self.detect_mode=='mix':
            input("LLM Mode without Lazy Detection will be very expensive! Please confirm... ")

        input("Press ENTER to continue...")
        if not os.path.exists(f"../output/{self.dataset}/"):
            os.makedirs(f"../output/{self.dataset}")
        if "project_name" in configs:
            self.project_name = configs["project_name"]
        else:
            self.project_name = generate_random_string(f"../output/{self.dataset}/number.txt")
        self.output_path = f"../output/{self.dataset}/{self.project_name}"
        create_paths(self.output_path)
        self.log_file = open(f"{self.output_path}/log.txt", 'w')
        import sys
        sys.stdout = Tee(self.log_file, sys.stdout)
        pprint.pprint(configs, sort_dicts=False)
        

    def build(self,  sequence_df,process_file = 'templates_krone_tree.csv'):

        self.graph = KroneTree(self.test_time_tracker, hardcode_kleene_pattern_summary=self.hardcode_kleene_pattern_summary,)
        process_path = f"../output/{self.dataset}/{process_file}"
        # process_path = f"../output/{self.dataset}/templates_krone_tree.csv"
        entity_col = 'entity_1'  # entity after filling
        action_col = 'action_1'  # action after filling
        structured_processes = pd.read_csv(process_path)
        structured_processes["event_id"] = structured_processes["event_id"].astype(str)
        self.train, self.test_1, self.test_2 = self._train_test_split(sequence_df)
        print(f"Total training: {len(self.train)}, Unique: {len(self.train['EventSequence'].unique())}")
        unique_count = len(
            set(self.test_1['EventSequence'].unique()).union(self.test_2['EventSequence'].unique())
        )
        print(f"Total testing: {len(self.test_1)+len(self.test_2)}, Unique: {unique_count}")
        sequences = self.train["EventSequence"].map(lambda s: [ str(e) for e in ast.literal_eval(s)]).tolist()
        seq_ids = self.train["seq_id"].tolist()

        self.graph.construct(structured_processes, entity_col, action_col)
        structured_process, modified = self.graph.node_detection(structured_processes)
        if modified:
            structured_process.to_csv(process_path)
        self.train_status_nodes = self.graph.inject_sequences(sequences, seq_ids)



    def _train_test_split(self, sequence_df, ):
        print("Splitting training and test data...")
        size = len(sequence_df)
        random_state = 0
        if "seq_id" not in sequence_df.columns:
            sequence_df["seq_id"] = sequence_df.index

        train_size = int(self.train_percent * size / 100)
        test_size = size - train_size

        test_1_size = int(self.test_1_percent * test_size / 100)

        train = sequence_df[sequence_df["Label"] == 0].sample(n=train_size, random_state=random_state).reset_index(
            drop=True)
        test = sequence_df[~sequence_df["seq_id"].isin(train["seq_id"].tolist())].reset_index(drop=True)
        test_1_size = test_1_size if test_1_size<=len(test) else len(test)
        test_1 = test.sample(n=test_1_size, random_state=random_state).reset_index(drop=True)
        test_1 = test_1.sort_values(by=["seq_id"])
        test_2 = test[~test["seq_id"].isin(test_1["seq_id"].tolist())].reset_index(drop=True)
        test_2 = test_2.sort_values(by=["seq_id"])

        print(f"Total data: {size}, training data: {train_size}, test 1 data: {test_1_size}, test 2 data: {len(test_2)}.")
        return train, test_1, test_2

    def _training_prepare(self):
        hist_train_knowledge = None
        self.curr_train_knowledge = None
        if self.dummy_summarize or self.dummy_detect:
            return self.curr_train_knowledge

        if self.detect_mode != 'local':
            if os.path.exists(f"../output/{self.dataset}/train_knowledge_all.csv"):
                hist_train_knowledge = pd.read_csv(f"../output/{self.dataset}/train_knowledge_all.csv")

            if hist_train_knowledge is not None:
                hist_train_knowledge = self._select_hist_train_knowledge(self.train, hist_train_knowledge)
            self.curr_train_knowledge = self.graph.knowledgeBase.load_or_generate_train_knowledge(
                knowledge_df=hist_train_knowledge, store_path = f"{self.output_path}/train_knowledge.csv")

            updated_hist_train_knowledge = self.update_hist_training_knowledge(hist_train_knowledge)
            updated_hist_train_knowledge.to_csv(f"../output/{self.dataset}/train_knowledge_all.csv", index=False)

        return self.curr_train_knowledge

    def update_hist_training_knowledge(self, train_knowlege_all=None):

        new_train_knowledge_all = pd.DataFrame()
        if self.detect_mode != "local":
            for level in ["STATUS", "ACTION", "ENTITY"]:
                if train_knowlege_all is not None:
                    train_knowlege_all_level = train_knowlege_all[train_knowlege_all["path_layer"] == level]
                    new_train_knowledge_level = self.curr_train_knowledge[
                        self.curr_train_knowledge["path_layer"] == level]
                    newnew_train = new_train_knowledge_level[~new_train_knowledge_level["overall_identifier"].isin(
                        train_knowlege_all_level["overall_identifier"].tolist())]
                    new_train_knowledge_all = pd.concat(
                        [new_train_knowledge_all, train_knowlege_all_level, newnew_train],
                        ignore_index=True)
                    print(f"Adding {level} training knowledge: {len(newnew_train)}")
                else:
                    new_train_knowledge_all = pd.concat([new_train_knowledge_all, self.curr_train_knowledge[
                        self.curr_train_knowledge["path_layer"] == level]])

        return new_train_knowledge_all

    def _select_hist_train_knowledge(self, data, data_knowledge_df):
        train_ids = data["seq_id"].tolist()
        keep_indices = []
        seq_ids_list = []

        for index, row in data_knowledge_df.iterrows():
            seq_id = ast.literal_eval(row["seq_ids"])
            common_seq_ids = set(seq_id).intersection(set(train_ids))
            if len(common_seq_ids) > 0:
                keep_indices.append(index)
                seq_id_str = "[" + ','.join([str(id) for id in common_seq_ids]) + "]"
                seq_ids_list.append(seq_id_str)

        selected_data_knowledge = data_knowledge_df.loc[keep_indices]
        selected_data_knowledge["seq_ids"] = seq_ids_list
        print(f"selected {len(selected_data_knowledge)} historical training knowledge!")
        return selected_data_knowledge

    def _test_prepare(self):
        summary_path = f"../output/{self.dataset}/test_summary_all.csv"
        embedding_path = f"../output/{self.dataset}/test_embedding_all.csv"
        knowledge_path = f"../output/{self.dataset}/test_knowledge_all.csv"
        if os.path.exists(summary_path) and self.load_history_test_summary:  # load all embeddings and summaries,
            self.hist_test_summary = pd.read_csv(summary_path)
            self.graph.knowledgeBase.load_test_path_summary(self.hist_test_summary)
        if os.path.exists(embedding_path) and self.load_history_test_embedding:
            self.hist_test_embedding = pd.read_csv(embedding_path)
            self.graph.knowledgeBase.load_test_path_embedding(self.hist_test_embedding)
        if os.path.exists(knowledge_path) and self.load_history_test_knowledge:
            self.hist_test_knowledge =pd.read_csv(knowledge_path)
            selected_test_knowledge = self._select_test_knowledge(self.test_1, self.hist_test_knowledge) # select only knowledge for test 1
            self.graph.knowledgeBase.load_test_path_knowledge(selected_test_knowledge)


    def _select_test_knowledge(self, data, data_knowledge_df):
        test_ids = data["seq_id"].tolist()
        keep_indices = []
        seq_ids_list = []
        if not self.status_level:
            data_knowledge_df = data_knowledge_df[data_knowledge_df["path_layer"] != 'STATUS']
        if not self.action_level:
            data_knowledge_df = data_knowledge_df[data_knowledge_df["path_layer"] != 'ACTION']
        if not self.entity_level:
            data_knowledge_df = data_knowledge_df[data_knowledge_df["path_layer"] != 'ENTITY']

        for index, row in data_knowledge_df.iterrows():
            seq_id = ast.literal_eval(row["seq_ids"])
            common_seq_ids = set(seq_id).intersection(set(test_ids))
            if len(common_seq_ids) > 0:
                keep_indices.append(index)
                seq_id_str = "[" + ','.join([str(id) for id in common_seq_ids]) + "]"
                seq_ids_list.append(seq_id_str)

        selected_data_knowledge = data_knowledge_df.loc[keep_indices]
        selected_data_knowledge["seq_ids"] = seq_ids_list
        print(f"selected {len(selected_data_knowledge)} historical test knowledge!")
        return selected_data_knowledge

    def run(self, test_2 = True, outputfile = False):

        if not os.path.exists(f"../output/{self.dataset}"):
            os.makedirs(f"../output/{self.dataset}")

        # training preparing
        self._training_prepare()
        # test preparing
        if self.detect_mode != 'local':
            self._test_prepare()

        test_1_mode = 'local' if self.detect_mode == 'local' else 'llm'
        test_2_mode = 'local' if self.detect_mode == 'local' else 'knowledge'
        # test part 1
        (test_results_1, llm_call_stats_1, test_1_local_precision, test_1_local_recall, test_1_local_f1,
         test_1_llm_precision, test_1_llm_recall, test_1_llm_f1, total_llm_requests, llm_summary,
         llm_detection) = self._test_1( test_1_mode)
        test_results = test_results_1

        test1_status_seqs = len(self.graph.knowledgeBase.non_GT_status_path_manager.paths)
        test1_action_seqs = len(self.graph.knowledgeBase.non_GT_action_path_manager.paths)
        test1_entity_seqs = len(self.graph.knowledgeBase.non_GT_entity_path_manager.paths)

        if test_2:
            # test part 2
            if len(self.test_2) > 0:
                print("=========== Starting Local detection ==========")
                test_results_2 = self._test_2(test_2_mode)
                test_results = pd.concat([test_results, test_results_2])

            # metric calculation
            (whole_local_precision, whole_local_recall, whole_local_f1,
             whole_llm_precision, whole_llm_recall, whole_llm_f1) = self.graph.detect_metrics(test_results,
                                                                                              self.entity_level,
                                                                                              self.action_level,
                                                                                              self.status_level)

            # write exp record
            self._write_exp(test_1_local_precision, test_1_local_recall, test_1_local_f1,
             test_1_llm_precision, test_1_llm_recall, test_1_llm_f1, total_llm_requests, llm_summary,
             llm_detection, whole_local_precision, whole_local_recall, whole_local_f1,
             whole_llm_precision, whole_llm_recall, whole_llm_f1)

        print(f"Status seqs in training knowledge: {len(self.graph.knowledgeBase.GT_status_path_manager.paths)}")
        print(f"Action seqs in training knowledge: {len(self.graph.knowledgeBase.GT_action_path_manager.paths)}")
        print(f"Entity seqs in training knowledge: {len(self.graph.knowledgeBase.GT_entity_path_manager.paths)}")
        print(f"Status seqs (uniqiue in level) in training knowledge: {len(self.graph.knowledgeBase.GT_status_path_manager.status_identifier_to_paths.keys())}")
        print(f"Action seqs (uniqiue in level) in training knowledge: {len(self.graph.knowledgeBase.GT_action_path_manager.action_identifier_to_paths.keys())}")
        print(f"Entity seqs (uniqiue in level) in training knowledge: {len(self.graph.knowledgeBase.GT_entity_path_manager.entity_identifier_to_paths.keys())}")
        print(f"Train sequences: {len(self.train)}, unique sequences: {len(self.train['EventSequence'].unique())}")

        print(f"Status seqs in testing knowledge: {len(self.graph.knowledgeBase.non_GT_status_path_manager.paths)}")
        print(f"Action seqs in testing knowledge: {len(self.graph.knowledgeBase.non_GT_action_path_manager.paths)}")
        print(f"Entity seqs in testing knowledge: {len(self.graph.knowledgeBase.non_GT_entity_path_manager.paths)}")
        print(f"Status seqs (uniqiue in level) in testing knowledge: {len(self.graph.knowledgeBase.non_GT_status_path_manager.status_identifier_to_paths.keys())}")
        print(f"Action seqs (uniqiue in level) in testing knowledge: {len(self.graph.knowledgeBase.non_GT_action_path_manager.action_identifier_to_paths.keys())}")
        print(f"Entity seqs (uniqiue in level) in testing knowledge: {len(self.graph.knowledgeBase.non_GT_entity_path_manager.entity_identifier_to_paths.keys())}")
        print(f"Test sequences in LLM test 1: {len(self.test_1)}, unique sequences: {len(self.test_1['EventSequence'].unique())}")
        self.test_time_tracker.report()
        #
        # print(f"Status seqs in testing knowledge: {test1_status_seqs}")
        # print(f"Action seqs in testing knowledge: {test1_action_seqs}")
        # print(f"Entity seqs in testing knowledge: {test1_entity_seqs}")
        # print(f"Test sequences in LLM test 1: {len(self.test_1)}, unique sequences: {len(self.test_1['EventSequence'].unique())}")

        self.log_file.close()



    def _test_1(self,test_1_mode):
        current_time = datetime.now()
        print(current_time)


        print("=========== Starting LLM detection ==========")
        (test_results_1, llm_call_stats_1, test_1_local_precision, test_1_local_recall, test_1_local_f1,
         test_1_llm_precision, test_1_llm_recall, test_1_llm_f1, krone_detection_res_viz, krone_decompose_res_viz) = self.graph.detect(self.test_1,
                                                                                     entity_level=self.entity_level,
                                                                                     action_level=self.action_level,
                                                                                     status_level=self.status_level,
                                                                                     detect_mode_status=test_1_mode,
                                                                                     detect_mode_action=test_1_mode,
                                                                                     detect_mode_entity=test_1_mode,
                                                                                     k_neighbors=self.k_neighbors,
                                                                                     automaton_adjustment=self.automaton_adjustment,
                                                                                     edge_consecutive_sensitive=self.edge_consecutive_sensitive,
                                                                                     dummy_summarize= self.dummy_summarize,
                                                                                     dummy_detect = self.dummy_detect,
                                                                                     lazy_detect = self.lazy_detect)
        print(f"Dummy summary in test 1: {self.graph.knowledgeBase.dummy_llm_summary_calls}")
        print(f"Dummy detect in test 1: {self.graph.knowledgeBase.dummy_llm_detection_calls}")

        status_path_embedding, status_path_summary, status_path_knowledge = (
            self.graph.knowledgeBase.non_GT_status_path_manager.store_path_embedding_summary_and_knowledge())
        action_path_embedding, action_path_summary, action_path_knowledge = (
            self.graph.knowledgeBase.non_GT_action_path_manager.store_path_embedding_summary_and_knowledge())
        entity_path_embedding, _, entity_path_knowledge = (
            self.graph.knowledgeBase.non_GT_entity_path_manager.store_path_embedding_summary_and_knowledge())

        path_embeddings = pd.concat([status_path_embedding, action_path_embedding, entity_path_embedding])
        path_summary = pd.concat([status_path_summary, action_path_summary])
        path_knowledge = pd.concat([status_path_knowledge, action_path_knowledge, entity_path_knowledge])

        path_embeddings.to_csv(f"{self.output_path}/test_embedding.csv", index=False)
        path_summary.to_csv(f"{self.output_path}/test_summary.csv", index=False)
        path_knowledge.to_csv(f"{self.output_path}/test_knowledge.csv", index=False)

        llm_call_stats_1 = pd.DataFrame(llm_call_stats_1)
        llm_call_stats_1.to_csv(f"{self.output_path}/llm_calls.csv", index=False)

        # self.krone_hierarchy.save(f"{self.output_path}/krone_hierarchy.pkl")
        total_llm_requests,llm_summary,llm_detection = self.graph.knowledgeBase.llm.store_call_details(f"{self.output_path}/llm_calls_detail.csv")

        test_results = pd.DataFrame(test_results_1)
        test_results.to_csv(f"{self.output_path}/test_results_1.csv", index=False)

        print(f"non GT status paths:{len(self.graph.knowledgeBase.non_GT_status_path_manager.paths)}")
        print(f"non GT action paths:{len(self.graph.knowledgeBase.non_GT_action_path_manager.paths)}")
        print(f"non GT entity paths:{len(self.graph.knowledgeBase.non_GT_entity_path_manager.paths)}")

        # storing into historical files
        if self.detect_mode!='local':
            embedding_file_path = f"../output/{self.dataset}/test_embedding_all.csv"
            if self.hist_test_embedding is not None:
                new_embedding = path_embeddings[~path_embeddings["overall_identifier"].isin(self.hist_test_embedding["overall_identifier"].tolist())]
                self.hist_test_embedding = pd.concat([self.hist_test_embedding, new_embedding])
                if len(new_embedding) >0:
                    file_path = f"../output/{self.dataset}/test_embedding_all.csv"
                    print(f"Adding {len(new_embedding)} to historical path embeddings: {embedding_file_path}")
                    self.hist_test_embedding.to_csv(file_path, index = False)
            else:
                self.hist_test_embedding = path_embeddings
                self.hist_test_embedding.to_csv(embedding_file_path, index=False)
                print(f"Adding {len(path_embeddings)} to historical path embeddings: {embedding_file_path}")

            if self.store_test_knowledge_as_history:
                summary_file_path = f"../output/{self.dataset}/test_summary_all.csv"

                knowledge_file_path = f"../output/{self.dataset}/test_knowledge_all.csv"
                if self.hist_test_summary is not None:
                    new_summary = path_summary[~path_summary["overall_identifier"].isin(self.hist_test_summary["overall_identifier"].tolist())]
                    self.hist_test_summary = pd.concat([self.hist_test_summary, new_summary])
                    if len(new_summary) > 0:
                        print(f"Adding {len(new_summary)} to historical path summaries: {summary_file_path}")
                        self.hist_test_summary.to_csv(summary_file_path, index=False)
                else:
                    self.hist_test_summary = path_summary
                    self.hist_test_summary.to_csv(summary_file_path, index=False)
                    print(f"Adding {len(path_summary)} to historical path summaries: {summary_file_path}")

                if self.hist_test_knowledge is not None:
                    new_knowledge = path_knowledge[~path_knowledge["overall_identifier"].isin(self.hist_test_knowledge["overall_identifier"].tolist())]
                    self.hist_test_knowledge = pd.concat([self.hist_test_knowledge, new_knowledge])
                    if len(new_knowledge) > 0:
                        file_path = f"../output/{self.dataset}/test_knowledge_all.csv"
                        print(f"Adding {len(new_knowledge)} to historical path knowledge: {knowledge_file_path}")
                        self.hist_test_knowledge.to_csv(file_path, index = False)
                else:
                    self.hist_test_knowledge = path_knowledge
                    self.hist_test_knowledge.to_csv(knowledge_file_path, index=False)
                    print(f"Adding {len(path_knowledge)} to historical path knowledge: {knowledge_file_path}")

        krone_detection_res_viz = pd.DataFrame(krone_detection_res_viz)
        krone_detection_res_viz.to_csv( f"../output/{self.dataset}/krone_viz_detection_res.csv")
        print(f"Sequence Detection res data stored in ../output/{self.dataset}/krone_viz_detection_res.csv!")

        krone_decompose_res_viz = pd.DataFrame(krone_decompose_res_viz)
        krone_decompose_res_viz.to_csv( f"../output/{self.dataset}/krone_decompose_res_viz.csv")
        print(f"Sequence decomposed data stored in ../output/{self.dataset}/krone_decompose_res_viz.csv!")

        return (test_results_1, llm_call_stats_1, test_1_local_precision, test_1_local_recall, test_1_local_f1,
         test_1_llm_precision, test_1_llm_recall, test_1_llm_f1, total_llm_requests,llm_summary,llm_detection)

    def _test_2(self, test_2_mode ):
        if len(self.test_2) > 0:
            print("=========== Starting Local detection ==========")
            test_results_2, _, _, _, _, _, _, _,_,_ = self.graph.detect(self.test_2, entity_level=self.entity_level, action_level=self.action_level, status_level=self.status_level,
                                                  detect_mode_status=test_2_mode, detect_mode_action = test_2_mode,
                                                  detect_mode_entity=test_2_mode,
                                                  k_neighbors=self.k_neighbors,
                                                  automaton_adjustment=self.automaton_adjustment,
                                                  edge_consecutive_sensitive=self.edge_consecutive_sensitive,
                                                                    lazy_detect = self.lazy_detect)
            test_results_2 = pd.DataFrame(test_results_2)
            test_results_2.to_csv(f"{self.output_path}/test_results_2.csv", index=False)
        else:
            test_results_2 = None

        return test_results_2


    def _write_exp(self,  test_1_local_precision, test_1_local_recall, test_1_local_f1,
         test_1_llm_precision, test_1_llm_recall, test_1_llm_f1, total_llm_requests, llm_summary,
         llm_detection, whole_local_precision, whole_local_recall, whole_local_f1,
         whole_llm_precision, whole_llm_recall, whole_llm_f1):

        import sys

        sys.stdout = sys.__stdout__

        if not os.path.exists(f"../output/{self.dataset}/exp.csv"):
            with open(f"../output/{self.dataset}/exp.csv", 'w') as f:
                f.write("exp,dataset,status,action,entity,train_percent,test_1_percent,detect_mode,k_neighbors,"
                        "test_1_local_precision,test_1_local_recall,test_1_local_f1,test_1_llm_precision,test_1_llm_recall,test_1_llm_f1,"
                        "whole_local_precision,whole_local_recall,whole_local_f1,whole_llm_precision,whole_llm_recall,whole_llm_f1,llm_total,llm_summary,llm_detection\n")

        with open(f"../output/{self.dataset}/exp.csv", 'a') as f:
            line = [self.project_name, self.dataset, self.status_level,self.action_level,self.entity_level,
                    self.train_percent, self.test_1_percent,
                    self.detect_mode, self.k_neighbors,
                    str(test_1_local_precision), test_1_local_recall, test_1_local_f1, test_1_llm_precision, test_1_llm_recall, test_1_llm_f1,
                    str(whole_local_precision), whole_local_recall, whole_local_f1, whole_llm_precision,whole_llm_recall, whole_llm_f1,
                    total_llm_requests, llm_summary, llm_detection
                    ]
            f.write(",".join([str(l) for l in line])+ '\n')




if __name__ == '__main__':
    pass