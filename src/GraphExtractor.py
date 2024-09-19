import asyncio
import nest_asyncio
nest_asyncio.apply()

from typing import Any, List, Callable, Optional, Union, Dict
from IPython.display import Markdown, display

from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import (
    default_parse_triplets_fn
)
from llama_index.core.graph_stores.types import (
    EntityNode, 
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    Relation
)
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import (
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
)
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core.bridge.pydantic import BaseModel, Field

class GraphRAGExtractor(TransformComponent):
    """
    Use an LLM and a single prompt + output to extract entity, relation from text.

    Args:
        llm (LLM): 
            extract model
        extract_prompt (Union[str, PromptTemplate]): 
            prompt to extract
        parse_fn (callable):
            a function to parse the output of language model
        num_workers (int):
            number of workers for processing nodes
        max_paths_per)chunks (int):
            max number of paths to extract per chunk
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunks: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunks: int = 10,
        num_workers: int=4
    ) -> None:
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)
        
        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunks=max_paths_per_chunks
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"
    
    def  __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs) # Goi phuong thuc acall
        )
    
    async def _aextract(self, node: BaseNode) -> BaseNode: 
        """Extract triples from a node"""
        assert hasattr(node, "text") # Kiem tra neu node co thuoc tinh text

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets = self.max_paths_per_chunks,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []
        
        # Lay ra cac gia tri da co
        existing_nodes = node.metadata.pop(KG_NODES_KEY, []) 
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata[
                "entity_description"
            ] = description
            entity_node = EntityNode(
                name = entity,
                label = entity_type,
                properties = metadata
            )
            existing_nodes.append(entity_node)
        
        metadata = node.metadata.copy()
        for triple in entities_relationship: # Lap qua cac triple
            subj ,rel, obj, description = triple # Lay cac gia tri
            subj_node = EntityNode(name=subj, properties = metadata) # Tao subj node
            obj_node = EntityNode(name=obj, properties=metadata) # Tao obj node
            metadata["relationship_description"] = description # Gan gia tri cho relationship_description
            rel_node = Relation( # Tao Relation
                label=rel,
                source_id=subj_node.id,
                target_id = obj_node.id,
                properties=metadata,
            )

            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
            self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples async"""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text"
        )