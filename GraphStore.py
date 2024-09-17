import os
from llama_index.llms.openai import OpenAI
llm = OpenAI(model='gpt4')

import re
from llama_index.core.graph_stores import SimplePropertyGraphStore
import networkx as nx
from graspologic.partition import hierarchical_leiden

from llama_index.core.llms import ChatMessage

class GraphRAGStore(SimplePropertyGraphStore):
    community_summary = {}
    max_cluster_size = 5

    def generate_community_summary(self, text):
        """Generate summary for a given text using LLM"""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", context=text)
        ]

        response = OpenAI().chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response
    
    def build_communities(self):
        """Build communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph() # Tao do thi NetworkX
        community_hierachical_clusters = hierarchical_leiden( # Ap dung thuat toan phan cum hierarchical_leiden
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        community_info = self._collect_community_info( # Thu thap thong tin chi tiet ve cong dong
            nx_graph, community_hierachical_clusters
        )
        self._summarize_communities(community_info) # Tom tat cong dong

    def _create_nx_graph(self):
        """Convert internal graph representation to NetworkX graph"""
        nx_graph = nx.Graph()
        for node in self.graph.nodes.values():
            nx_graph.add_node(str(node))

        for relation in self.graph.relations.values():
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph
    
    def _collect_community_info(self, nx_graph, clusters): 
        """Collect detailed information for each node based on their community"""
        community_mapping = {item.node: item.cluster for item in clusters} # Tao anh xa giua node va communities
        community_info = {}
        for item in clusters:
            cluster_id = item.cluster # Lay ID cua community
            node = item.node # Lay node
            if cluster_id not in community_info:
                community_info[cluster_id] = []

            for neighbor in nx_graph.neighbors(node):
                if community_mapping[neighbor] == cluster_id: # Kiem tra neu cac node lan can thuoc cung community
                    edge_data = nx_graph.get_edge_data(node, neighbor) # Lay du lieu edge
                    if edge_data:
                        detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                        community_info[cluster_id].append(detail) # Them chi tiet vao community_info
        
        return community_info
    
    def _summarize_communities(self, community_info): # Tao va luu tru cac ban tom tat cho tung community
        """Generate and store summaries for each community"""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "." # Dam bao ket thuc bang dau cham
            )
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text) # Tao ban tom tat va luu tru

    def get_community_summaries(self): # Tra ve cac ban tom tat community
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary: # Kiem tra neu chua co ban tom tat
            self.build_communities() # Xay dung cac community
        return self.community_summary 