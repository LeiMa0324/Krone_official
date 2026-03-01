from typing import Dict, List, Set, Tuple

node_type_structure = {"ROOT":"ENTITY", "ENTITY":"ACTION", "ACTION":"STATUS", "STATUS":"NONE"}
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def condense_path(node_path: list['Node']) -> list['Node']:
    '''remove the consecutive repeating events in the sequence, except for the status'''
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

def node_list_to_identifier_text(node_path: List['Node']) -> str:
    return ','.join([node.node_identifier for node in node_path])

def path_to_semantic_text(node_path: List['Node']) -> str:
    return ','.join([node.node_identifier.split('_')[0] for node in node_path])

class Node:

    def __init__(self, node_id: int, node_identifier: str, node_type: str, t_ids: Set[str]=None,  template = '', template_summary = '',
                 is_anomaly= False, is_anomaly_reason='', row_id= -1):
        self.row_id = row_id  # the row id in structured_process, only for status node
        self.node_id = node_id
        self.node_identifier = node_identifier  # the name of the node, as an indentifier with the id
        # self.node_identifier = node_name + '_' + node_id
        self.node_type = node_type # the type of the node, [ROOT, ENTITY, ACTION, STATUS]
        self.entity_type = 'none' # the entity_type if the node is an entity

        self.is_anomaly = is_anomaly
        self.is_anomaly_reason = is_anomaly_reason

        self.child_node_type = node_type_structure[self.node_type]
        self.parent: Node = None
        self.children: Dict[str, Node] = {}
        self.t_id_to_children: Dict[str, Node] = {}

        self.outgoing_neighbors: Dict[str, Node] = {}
        self.incoming_neighbors: Dict[str, Node] = {}

        # the templates that passing this node
        self.template_ids: Set[str] = t_ids
        self.template = template
        self.template_summary = template_summary

        # training paths under this node
        self.children_paths: Dict[str: List[Node]] = {}
        # identifier_seq_text to next nodes
        self.children_path_next: Dict[str: Dict[str: Node]] = {}
        self.children_path_seqs_summary: Dict[str: str] = {} # store the llm summaries for the children paths under this node

        self.unseen_paths: Dict[str: List[Node]] = {}
        self.unseen_path_count: Dict[str: int] = {}

        self.masked = False

    def strong_chain(self):
        chain = []
        left_visted = [self]
        left_chain = self.left_strong_chain(left_visted)
        right_visted = [self]
        right_chain = self.right_strong_chain(right_visted)
        chain.extend(left_chain)
        chain.append(self)
        chain.extend(right_chain)

        return chain

    def left_strong_chain(self, left_visted):
        chain = []
        if len(self.incoming_neighbors.values()) == 0 or len(self.incoming_neighbors.values()) > 1:
            return chain
        else:
            left_neighbor = list(self.incoming_neighbors.values())[0]
            if left_neighbor == self or len(left_neighbor.outgoing_neighbors.values()) > 1 :
                return chain # if the neighbor is itself or the left neighbor has multiple outgoings
            left = list(self.incoming_neighbors.values())[0]
            if left in left_visted:
                return chain

            chain = [left]
            left_visted.append(left)
            left_chain = left.left_strong_chain(left_visted)
            chain.extend(left_chain)
            return chain

    def right_strong_chain(self, right_visted):
        chain = []
        if len(self.outgoing_neighbors.values()) == 0 or len(self.outgoing_neighbors.values()) > 1:
            return chain
        else:
            right_neighbor = list(self.outgoing_neighbors.values())[0]
            if right_neighbor == self or len(right_neighbor.incoming_neighbors.values()) > 1:
                return chain
            right = list(self.outgoing_neighbors.values())[0]
            if right in right_visted:
                return chain

            chain = [right]
            right_visted.append(right)
            neihbor_chain = right.right_strong_chain(right_visted)
            chain.extend(neihbor_chain)
            return chain

    def get_semantic_node_name(self):
        return self.node_identifier.split('_')[0]

    def add_child(self, child: 'Node'):
        self.children[child.node_identifier] = child
        for t_id in child.template_ids:
            self.t_id_to_children[t_id] = child
        child.parent = self


    def add_outgoing_neighbor(self, neighbor: 'Node'):
        if neighbor.node_type == self.node_type:
            self.outgoing_neighbors[neighbor.node_identifier] = neighbor
            neighbor.incoming_neighbors[self.node_identifier] = self

    def find_entity_node(self):
        if self.node_type == 'ENTITY':
            return self
        elif self.node_type == 'ACTION':
            return self.parent
        elif self.node_type == 'STATUS':
            return self.parent.parent
        else:
            raise NotImplemented("Invalid node type {self.node_type}")


    def find_nodes_for_t_id(self, t_id: str, nodes: List['Node']):
        if t_id in self.t_id_to_children.keys():
            child = self.t_id_to_children[t_id]
            nodes.append(child)
            nodes = child.find_nodes_for_t_id(t_id, nodes)
            return nodes
        else:
            return nodes

    def add_path_in_children(self, children_path: List['Node'], next_node: 'Node' = None):
        condensed_path = condense_path(children_path)
        identifier_seq = node_list_to_identifier_text(condensed_path)
        if identifier_seq not in self.children_paths.keys():
            self.children_paths[identifier_seq] = condensed_path
        if next_node is not None:
            next_node_dict = self.children_path_next.get(identifier_seq, {})
            next_node_dict[next_node.node_identifier] = next_node
            self.children_path_next[identifier_seq] = next_node_dict

    def add_unseen_path_local(self, children_path: List['Node']):
        condensed_path = condense_path(children_path)
        identifier_seq = node_list_to_identifier_text(condensed_path)

        if identifier_seq not in self.unseen_paths.keys():
            self.unseen_paths[identifier_seq] = condensed_path
            count = self.unseen_path_count.get(identifier_seq, 0)
            self.unseen_path_count[identifier_seq] = count + 1


    def find_similar_paths(self, node_path: List['Node'], k = 3, revectorize = True ) -> List[List['Node']]:
        # vectorize all stored paths
        # Initialize the TF-IDF Vectorizer
        if len(self.children_paths.keys()) ==0:
            return []
        vectorizer = TfidfVectorizer()
        seqeunces = []
        raw_sequences = []
        for path_seq in self.children_paths.keys():
            raw_sequences.append(path_seq)
            seq_list = path_seq.split(",")
            new_seq_list = []
            for event in seq_list:
                new_seq_list.append(event.split("_")[0])
            seqeunces.append(" ".join(new_seq_list))

        self.tfidf_matrix = vectorizer.fit_transform(seqeunces)
        self.vectorizer = vectorizer

        seq = path_to_semantic_text(node_path)
        test_seq = " ".join(seq)
        tfidf_test_vector = self.vectorizer.transform([test_seq])
        cosine_sim = cosine_similarity(tfidf_test_vector, self.tfidf_matrix)
        top_k_indices = cosine_sim[0].argsort()[-k:][::-1]  # Get indices of top k similarities
        top_k_similarities = cosine_sim[0][top_k_indices]

        raw_top_k_sequences = [raw_sequences[i] for i in top_k_indices]
        top_k_node_paths = [self.children_paths[s] for s in raw_top_k_sequences]

        return top_k_node_paths

    def add_template(self, template: str, template_summary: str):
        self.template = template
        self.template_summary = template_summary

