import time

from krone_hierarchy.Krone_seq import *
from krone_hierarchy.Automaton_graph import *

class Path_manager():

    def __init__(self, level: str, if_GT: bool, llm: LLM):
        self.llm = llm
        self.if_GT = if_GT
        self.level = level
        self.paths: Dict[str, KroneSeq] = {}
        self.entity_identifier_to_paths: Dict[str, List[KroneSeq]] = {}
        self.action_identifier_to_paths: Dict[str, List[KroneSeq]] = {}
        self.status_identifier_to_paths: Dict[str, List[KroneSeq]] = {}

        self.llm_knowledge_df = pd.DataFrame()
        self.embedding_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()

        self.status_identifier_lists = None
        self.action_identifier_lists = None
        self.entity_identifier_lists = None

        self.valid_paths = None
        self.valid_path_embeddings = None

        self.automaton_graph = Automaton_graph()

    def generate_temp_path(self, node_list: List[Node], seq_id: int, children_paths: List[KroneSeq] = None, if_GT=False):
        condensed_node_list = condense_path(node_list)
        temp_path = KroneSeq(node_list=condensed_node_list,
                             parent_node=condensed_node_list[0].parent,
                             seq_id=seq_id, children_paths=children_paths, if_GT=if_GT)

        temp_path.uncollapsed_no_list = node_list

        return temp_path

    def has_path_sliding(self, path: KroneSeq, identifier_level) -> bool:

        if identifier_level == 'STATUS':
            if self.status_identifier_lists is None:
                self.status_identifier_lists = [key.split(',') for key in self.status_identifier_to_paths.keys()]
            test_identifier = path.status_identifier.split(',')
            sliding_coverage = contains_existing_seq(test_identifier, self.status_identifier_lists)
        elif identifier_level == 'ACTION':
            if self.action_identifier_lists is None:
                self.action_identifier_lists = [key.split(',') for key in self.action_identifier_to_paths.keys()]
            test_identifier = path.action_identifier.split(',')
            sliding_coverage = contains_existing_seq(test_identifier, self.action_identifier_lists)
        elif identifier_level == 'ENTITY':
            if self.entity_identifier_lists is None:
                self.entity_identifier_lists = [key.split(',') for key in self.entity_identifier_to_paths.keys()]
            test_identifier = path.entity_identifier.split(',')
            sliding_coverage = contains_existing_seq(test_identifier, self.entity_identifier_lists)
        else:
            raise NotImplementedError(f"{self.level} level not supported for _generate_action_identifier!")

        return sliding_coverage

    def has_path(self, path: KroneSeq, identifier_level=None) -> bool:
        if identifier_level is None:
            identifier_level = 'overall'

        if identifier_level == 'overall':
            return path.overall_identifier in self.paths.keys()
        elif identifier_level == 'STATUS':
            return path.status_identifier in self.status_identifier_to_paths.keys()
        elif identifier_level == 'ACTION':
            return path.action_identifier in self.action_identifier_to_paths.keys()
        elif identifier_level == 'ENTITY':
            return path.entity_identifier in self.entity_identifier_to_paths.keys()
        else:
            raise NotImplementedError(f"{self.level} level not supported for _generate_action_identifier!")

    def get_path(self, overall_identifier: str) -> KroneSeq:
        return self.paths[overall_identifier]

    def add_path(self, temp_path, seq_id, maintain_automaton = False):

        if temp_path.overall_identifier not in self.paths.keys():
            self.paths[temp_path.overall_identifier] = temp_path
            s_paths = self.status_identifier_to_paths.get(temp_path.status_identifier, [])
            s_paths.append(temp_path)
            self.status_identifier_to_paths[temp_path.status_identifier] = s_paths

            a_paths = self.action_identifier_to_paths.get(temp_path.action_identifier, [])
            a_paths.append(temp_path)
            self.action_identifier_to_paths[temp_path.action_identifier] = a_paths

            e_paths = self.entity_identifier_to_paths.get(temp_path.entity_identifier, [])
            e_paths.append(temp_path)
            self.entity_identifier_to_paths[temp_path.entity_identifier] = e_paths
            path = temp_path
        else:
            path = self.paths[temp_path.overall_identifier]

        path.add_sequence_id(seq_id=seq_id)
        if maintain_automaton:
            self.automaton_graph.add_path(path)

        return path

    def remove_path(self, path: KroneSeq):
        if path.overall_identifier in self.paths.keys():
            del self.paths[path.overall_identifier]
        if path.status_identifier in self.status_identifier_to_paths.keys():
            del self.status_identifier_to_paths[path.status_identifier]
        if path.action_identifier in self.action_identifier_to_paths.keys():
            del self.action_identifier_to_paths[path.action_identifier]
        if path.entity_identifier in self.entity_identifier_to_paths.keys():
            del self.entity_identifier_to_paths[path.entity_identifier]

    def find_similar_paths_by_embedding(self, test_path: KroneSeq, k=3) -> Tuple[List[KroneSeq], List[float]]:
        all_paths = self.find_paths_under_same_parent(test_path)
        # only allow local predicted normal_flows patterns as examples
        all_normal_paths = [path for path in all_paths if (path.path_pred == 0) and (
                    path.path_pred_reason in [LOCAL_DETECTOR_NORMAL_REASON, TRAIN_NORMAL_REASON])]
        non_empty_paths = [path for path in all_normal_paths if not path.is_empty_path]
        n_paths = []
        top_scores = []

        if len(non_empty_paths) > 0:
            embeddings = []
            for path in non_empty_paths:
                pattern_embedding = path.generated_pattern_embedding()
                embeddings.append(pattern_embedding)

            embeddings = torch.stack(embeddings)
            test_embedding = test_path.generated_pattern_embedding()
            scores = util.pytorch_cos_sim(test_embedding, embeddings)[0]
            top_results = torch.topk(scores, k=min(k, len(scores)))

            top_scores = top_results[0].tolist()
            indices = top_results[1].tolist()
            n_paths = [non_empty_paths[index] for index in indices]
            del embeddings

        return n_paths, top_scores

    # def find_alternative_paths_by_start_and_end_nodes(self, test_path: Path, k = 3) -> Tuple[List[Path],List[float]]:
    #     # use bfs to find the k alternative paths from start to end node
    #     start_node = test_path.node_list[0]
    #     end_node = test_path.node_list[-1]
    #
    #     traversed = []
    #     for s

    def find_paths_under_same_parent(self, path) -> List[KroneSeq]:
        level = path.level
        if level == 'STATUS':
            if path.action_identifier in self.action_identifier_to_paths.keys():
                return self.action_identifier_to_paths[path.action_identifier]
            else:
                return []
        elif level == 'ACTION':
            if path.entity_identifier in self.entity_identifier_to_paths.keys():
                return self.entity_identifier_to_paths[path.entity_identifier]
            else:
                return []
        elif level == 'ENTITY':
            return list(self.paths.values())
        else:
            raise NotImplementedError(f"Level {level} not supported for find_paths_under_same_parent!")

    def load_path_embedding_for_path(self, path: KroneSeq):

        if_load_embedding = (len(self.embedding_df) > 0 and path.pattern_embedding is None
                             and path.overall_identifier in self.embedding_df["overall_identifier"].tolist()
                             )
        if if_load_embedding:
            row = \
            self.embedding_df[self.embedding_df["overall_identifier"] == path.overall_identifier].iloc[
                0]
            if isinstance(row["pattern_embedding"], float):
                embedding = None
            else:
                print("Loading test embedding")
                embedding = [float(f) for f in
                             row["pattern_embedding"].replace("[", "").replace("]", "").split(",")]
                embedding = torch.tensor(embedding)

            path.pattern_embedding = embedding

    def load_path_summary_for_path(self, path: KroneSeq):

        if_load_summary = (len(self.summary_df) > 0 and path.path_summary == INITIAL_BLANK_SUMMARY
                           and path.overall_identifier in self.summary_df["overall_identifier"].tolist()
                           )
        path_summary = None
        if if_load_summary:
            row = \
            self.summary_df[self.summary_df["overall_identifier"] == path.overall_identifier].iloc[
                0]
            if isinstance(row["path_summary"], float):
                path_summary = None
            else:
                print("Loading test summary")
                path_summary = row["path_summary"]

            path.path_summary = path_summary
        return path_summary

    def load_path_prediction_for_path(self, path: KroneSeq):

        if len(self.llm_knowledge_df) > 0 and path.overall_identifier in self.llm_knowledge_df["overall_identifier"].tolist():
            row = self.llm_knowledge_df[self.llm_knowledge_df["overall_identifier"] == path.overall_identifier]
            assert len(row) == 1
            row = row.iloc[0]

            if "path_reason" in self.llm_knowledge_df.columns:
                if isinstance(row["path_reason"], float):
                    row["path_reason"] = INITIAL_BLANK_SUMMARY
                path.path_pred_reason = row["path_reason"]

            if "path_pred" in self.llm_knowledge_df.columns:
                path.path_pred = int(row["path_pred"])

            if "seq_ids" in self.llm_knowledge_df.columns:
                for id in ast.literal_eval(row["seq_ids"]):
                    path.add_sequence_id(id)

    def store_path_embedding_summary_and_knowledge(self):
        path_embedding = {"path_layer": [], "seq_ids": [], "entity_identifier": [], "action_identifier": [],
                          "status_identifier": [], "overall_identifier": [], "logkey_seq": [],
                          "pattern_embedding": [],
                          "is_empty_path": [], "if_GT":[]}
        path_summary = {"path_layer": [], "seq_ids": [], "entity_identifier": [], "action_identifier": [],
                          "status_identifier": [], "overall_identifier": [], "logkey_seq": [],
                          "path_summary": [],
                        "is_empty_path": [], "if_GT": []}
        path_knowledge = {"path_layer": [], "seq_ids": [], "entity_identifier": [], "action_identifier": [],
                          "status_identifier": [], "overall_identifier": [], "logkey_seq": [],
                          "path_pred": [], "path_reason": [],
                          "is_empty_path": [], "if_GT": []}

        for identifier, path in self.paths.items():
            if path.pattern_embedding is not None:
                path_embedding["path_layer"].append(path.level)
                path_embedding["seq_ids"].append("[" + ",".join([str(id) for id in list(path.sequence_ids)]) + "]")
                path_embedding["entity_identifier"].append(path.entity_identifier)
                path_embedding["action_identifier"].append(path.action_identifier)
                path_embedding["status_identifier"].append(path.status_identifier)
                path_embedding["overall_identifier"].append(path.overall_identifier)
                path_embedding["logkey_seq"].append(path.find_log_key_seq())
                path_embedding["pattern_embedding"].append(path.pattern_embedding.tolist())
                path_embedding["is_empty_path"].append(path.is_empty_path)
                path_embedding["if_GT"].append(self.if_GT)

            if path.path_summary!= INITIAL_BLANK_SUMMARY:
                path_summary["path_layer"].append(path.level)
                path_summary["seq_ids"].append("[" + ",".join([str(id) for id in list(path.sequence_ids)]) + "]")
                path_summary["entity_identifier"].append(path.entity_identifier)
                path_summary["action_identifier"].append(path.action_identifier)
                path_summary["status_identifier"].append(path.status_identifier)
                path_summary["overall_identifier"].append(path.overall_identifier)
                path_summary["logkey_seq"].append(path.find_log_key_seq())
                path_summary["path_summary"].append(path.path_summary)
                path_summary["is_empty_path"].append(path.is_empty_path)
                path_summary["if_GT"].append(self.if_GT)

            if path.path_pred !=-1 and path.llm_predicted == True:
                path_knowledge["path_layer"].append(path.level)
                path_knowledge["seq_ids"].append("[" + ",".join([str(id) for id in list(path.sequence_ids)]) + "]")
                path_knowledge["entity_identifier"].append(path.entity_identifier)
                path_knowledge["action_identifier"].append(path.action_identifier)
                path_knowledge["status_identifier"].append(path.status_identifier)
                path_knowledge["overall_identifier"].append(path.overall_identifier)
                path_knowledge["logkey_seq"].append(path.find_log_key_seq())
                path_knowledge["path_pred"].append(path.path_pred)
                path_knowledge["path_reason"].append(path.path_pred_reason)
                path_knowledge["is_empty_path"].append(path.is_empty_path)
                path_knowledge["if_GT"].append(self.if_GT)


        path_knowledge = pd.DataFrame(path_knowledge)
        path_embedding = pd.DataFrame(path_embedding)
        path_summary = pd.DataFrame(path_summary)

        return path_embedding, path_summary, path_knowledge
    def store_training_path_knowledge(self):
        path_knowledge = {"path_layer": [], "seq_ids": [], "entity_identifier": [], "action_identifier": [],
                          "status_identifier": [], "overall_identifier": [], "logkey_seq": [], "path_summary": [],
                          "path_pred": [], "path_reason": [],
                          "pattern_embedding": [], "is_empty_path": []}
        for identifier, path in self.paths.items():
            path_knowledge["path_layer"].append(path.level)
            path_knowledge["seq_ids"].append("[" + ",".join([str(id) for id in list(path.sequence_ids)]) + "]")
            path_knowledge["entity_identifier"].append(path.entity_identifier)
            path_knowledge["action_identifier"].append(path.action_identifier)
            path_knowledge["status_identifier"].append(path.status_identifier)
            path_knowledge["overall_identifier"].append(path.overall_identifier)
            path_knowledge["logkey_seq"].append(path.find_log_key_seq())
            path_knowledge["path_summary"].append(path.path_summary)
            path_knowledge["path_pred"].append(path.path_pred)
            path_knowledge["path_reason"].append(path.path_pred_reason)
            if path.pattern_embedding is not None:
                path_knowledge["pattern_embedding"].append(path.pattern_embedding.tolist())
            else:
                path_knowledge["pattern_embedding"].append(path.pattern_embedding)

            path_knowledge["is_empty_path"].append(path.is_empty_path)

        path_knowledge = pd.DataFrame(path_knowledge)
        path_knowledge["if_GT"] = self.if_GT
        return pd.DataFrame(path_knowledge)

    def summarize_GT_paths(self, hardcode_kleene_pattern_summary=False):
        print(f"========== Summarizing {self.level} Sequences... =========")
        count = 0
        created = False
        non_summarized_paths = [path for path in self.paths.values() if
                                (not path.is_empty_path) and path.path_summary == INITIAL_BLANK_SUMMARY]
        start = time.time()
        llm_summaries = 0
        for path in tqdm(non_summarized_paths):
            summary, llm_summary = path.summarize_path(self.llm, hardcode_kleene_pattern_summary)
            llm_summaries +=llm_summary
            created = True
            count += 1
        end = time.time()


        print(f"========== Summarizing {count} {self.level} Sequences Finished! Duration: {end - start}, LLM summary: {llm_summaries} =========")
        self.stage = f'TRAINING {self.level} PATH SUMMARIZATION'
        return created

    def batch_generate_embeddings(self, batch_size=32):
        created = False
        self.valid_paths = [
            path for path in self.paths.values()
            if path.pattern_embedding is None and not path.is_empty_path
        ]

        if len(self.valid_paths) > 0:
            print(f"========== Generating {self.level} Training Sequence Embeddings... =========")
            created = True

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer('bert-base-nli-mean-tokens')
            model = model.to(device)

            transformer_model = model._first_module().auto_model
            tokenizer = model.tokenizer  # Avoid redundant calls to model.tokenizer

            start = time.time()

            # Process sequences in batches
            for batch_start in tqdm(range(0, len(self.valid_paths), batch_size)):
                batch_paths = self.valid_paths[batch_start:batch_start + batch_size]
                sequences = [path.find_logkey_sequence_str() for path in batch_paths]

                # Tokenization
                inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

                # Move inputs to the appropriate device (CPU or GPU)
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Generate embeddings
                with torch.no_grad():
                    outputs = transformer_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        return_dict=True
                    )
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu()  # Ensure embeddings are in CPU memory

                # Assign embeddings to paths
                for i, path in enumerate(batch_paths):
                    path.pattern_embedding = batch_embeddings[i]

                # Free memory manually
                del inputs, outputs, batch_embeddings  # Delete large variables to free memory

            end = time.time()
            print(f"========== Finished Generating {len(self.valid_paths)} {self.level} Training Sequence Embeddings! Duration: {end - start:.2f} sec =========")

        return created
