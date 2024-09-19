from llama_index.core import PropertyGraphIndex
from llama_index.llms.openai import OpenAI
from typing import Any
import re

from src.GraphExtractor import GraphRAGExtractor
from src.GraphQueryEngine import GraphRAGQueryEngine
from src.GraphStore import GraphRAGStore

# Initialize the OpenAI model
llm = OpenAI(model="gpt4")


class GraphRetriever:
    def __init__(self, nodes, llm, **kwargs) -> None:
        self.nodes = nodes  # nodes build from raw documents
        self.llm = llm

    def run(self, query):
        KG_TRIPLET_EXTRACT_TMPL = """
            -Goal-
            Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
            Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

            -Steps-
            1. Identify all entities. For each identified entity, extract the following information:
            - entity_name: Name of the entity, capitalized
            - entity_type: Type of the entity
            - entity_description: Comprehensive description of the entity's attributes and activities
            Format each entity as ("entity"$$$$<entity_name>$$$$<entity_type>$$$$<entity_description>)

            2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
            For each pair of related entities, extract the following information:
            - source_entity: name of the source entity, as identified in step 1
            - target_entity: name of the target entity, as identified in step 1
            - relation: relationship between source_entity and target_entity
            - relationship_description: explanation as to why you think the source entity and the target entity are related to each other

            Format each relationship as ("relationship"$$$$<source_entity>$$$$<target_entity>$$$$<relation>$$$$<relationship_description>)

            3. When finished, output.

            -Real Data-
            ######################
            text: {text}
            ######################
            output:"""

        kg_extractor = GraphRAGExtractor(
            llm=self.llm,  # Sử dụng LLM đã được định nghĩa trước đó
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,  # Sử dụng prompt template để trích xuất các triplet
            max_paths_per_chunk=2,  # Đặt số lượng đường dẫn tối đa mỗi chunk là 2
            parse_fn=self.parse_fn,  # Sử dụng hàm parse_fn để phân tích kết quả trả về
        )

        index = PropertyGraphIndex(
            nodes=self.nodes,
            property_graph_store=GraphRAGStore(),
            kg_extractors=[kg_extractor],
            show_progress=True,
        )

        index.property_graph_store.build_communities()

        query_engine = GraphRAGQueryEngine(
            graph_store=index.property_graph_store, llm=self.llm
        )

        response = query_engine.query(query)

        return response

    def parse_fn(response_str: str) -> Any:

        # Pattern này sẽ khớp với một chuỗi có định dạng (entity$$$$value1$$$$value2$$$$value3), trong đó value1, value2, và value3 là các giá trị của entity
        entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
        # Pattern này sẽ khớp với một chuỗi có định dạng (relationship$$$$value1$$$$value2$$$$value3$$$$value4), trong đó value1, value2, value3, và value4 là các giá trị của relationship.
        relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'

        # Tìm tất cả các entities trong response_str sử dụng entity_pattern
        entities = re.findall(entity_pattern, response_str)

        # Tìm tất cả các relationships trong response_str sử dụng relationship_pattern
        relationships = re.findall(relationship_pattern, response_str)

        # Trả về danh sách các entities và relationships
        return entities, relationships


def load_data(path):
    from llama_index.core import SimpleDirectoryReader

    documents = ""
    return documents


if __name__ == "__main__":
    documents = load_data("path_to_data")
    from llama_index.core.text_splitter import TokenTextSplitter

    node_parser = TokenTextSplitter(chunk_size=256)

    nodes = node_parser.get_nodes_from_documents(documents=documents)
    query = "Đưa ra câu hỏi tương ứng bộ dữ liệu đầu vào"
    graphRetriever = GraphRetriever(nodes=nodes, llm=llm)
    graphRetriever.run(query=query)
