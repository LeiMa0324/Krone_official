from typing import Dict, List, Set, Tuple
from krone_hierarchy.Krone_seq import *
from krone_hierarchy.Node import *

class Automaton_node:

    def __init__(self, node_identifier):
        self.node_identifier = node_identifier
        self.incoming_edges: Dict[str: 'Automaton_edge'] = {} # prev node id, edge obj
        self.outgoing_edges: Dict[str: 'Automaton_edge'] = {} # next node id, edge obj

    def has_incoming_edge(self, prev_node: 'Automaton_node', seq_ids = None) -> bool:
        if prev_node.node_identifier in self.incoming_edges.keys():
            if seq_ids is None:
                return True
            else:
                edge = self.incoming_edges[prev_node.node_identifier]
                common_seq_id = seq_ids & edge.false_negative_seq_ids
                return len(common_seq_id) > 0
        else:
            return False


    def has_outgoing_edge(self, next_node: 'Automaton_node', seq_ids = None) -> bool:
        if next_node.node_identifier in self.outgoing_edges.keys():
            if seq_ids is None:
                return True
            else:
                edge = self.outgoing_edges[next_node.node_identifier]
                common_seq_id = seq_ids & edge.false_negative_seq_ids
                return len(common_seq_id) > 0
        else:
            return False


class Automaton_edge:

    def __init__(self, start_node: Automaton_node, end_node: Automaton_node):
        self.start = start_node
        self.end = end_node
        self.seq_ids = set()

        self.start.outgoing_edges[self.end.node_identifier] = self
        self.end.incoming_edges[self.start.node_identifier] = self



class Automaton_graph:

    def __init__(self):
        self.automaton_nodes: Dict[str: Automaton_node] = {}

    def add_path(self, path: KroneSeq):
        seq_ids = path.sequence_ids
        for i, k_node in enumerate(path.node_list):
            a_node = self.node_transfer(k_node)
            if i < len(path.node_list) -1:
                next_node = path.node_list[i+1]
                a_next_node = self.node_transfer(next_node)
                self.connect_nodes(a_node, a_next_node, seq_ids)

    def node_transfer(self, k_node: Node):
        if k_node.node_identifier in self.automaton_nodes.keys():
            a_node = self.automaton_nodes[k_node.node_identifier]
        else:
            a_node = Automaton_node(k_node.node_identifier)
            self.automaton_nodes[k_node.node_identifier] = a_node
        return a_node

    def connect_nodes(self, start: Automaton_node, end:Automaton_node, seq_ids):
        if end.has_incoming_edge(start):
            edge = end.incoming_edges[start.node_identifier]
        else:
            edge = Automaton_edge(start, end)
        edge.seq_ids.update(seq_ids)


    def path_diffs(self, path: KroneSeq, edge_consecutive_sensitive = False):
        results = []
        cur_diff_nodes = []
        cur_diff_children_paths = []
        if_diff = False
        last_edge = None
        for i, k_node in enumerate(path.node_list):
            a_node = self.node_transfer(k_node)
            if i < len(path.node_list) - 1:
                next_k_node = path.node_list[i + 1]
                next_a_node = self.node_transfer(next_k_node)

                connected = a_node.has_outgoing_edge(next_a_node)
                consecutive_edges = False
                if connected:
                    edge = a_node.outgoing_edges[next_a_node.node_identifier]
                    if last_edge:
                        consecutive_edges = len(edge.seq_ids & last_edge.seq_ids) > 0
                    else:
                        consecutive_edges = True
                    last_edge = edge
                else:
                    consecutive_edges = False
                    last_edge = None

                if not edge_consecutive_sensitive:
                    consecutive_edges = True  # ignore consecutive edges

                if if_diff:  # inside a diff segment
                    cur_diff_nodes.append(k_node)
                    if path.level in ['ACTION','ENTITY']:
                        cur_diff_children_paths.append(path.children_paths[i])
                    if connected and consecutive_edges:  # diff -> non_diff
                        result_tuple = (cur_diff_nodes, cur_diff_children_paths)
                        results.append(result_tuple)
                        cur_diff_nodes = []
                        cur_diff_children_paths = []
                        if_diff = False
                else:  # non-diff -> diff
                    if not (connected and consecutive_edges):  # diff start
                        cur_diff_nodes.append(k_node)
                        if path.level in ['ACTION', 'ENTITY']:
                            cur_diff_children_paths.append(path.children_paths[i])
                        if_diff = True

        return results

    def traverse_detect(self, path: KroneSeq):
        last_edge = None
        for i, k_node in enumerate(path.node_list):
            a_node = self.node_transfer(k_node)
            if i < len(path.node_list) - 1:
                next_k_node = path.node_list[i + 1]
                next_a_node = self.node_transfer(next_k_node)
                connected =  next_a_node.has_incoming_edge(a_node,)
                consecutive_edges = False
                if connected:
                    edge = a_node.outgoing_edges[next_a_node.node_identifier]
                    if last_edge:
                        consecutive_edges = len(edge.false_negative_seq_ids & last_edge.seq_id) > 0
                    else:
                        consecutive_edges = True

                if connected and consecutive_edges:
                    continue
                else:
                    return False

        return True

