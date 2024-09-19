from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM
from GraphStore import GraphRAGStore
from GraphExtractor import GraphRAGExtractor
from llama_index.core.llms import ChatMessage
import re


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answer to a specific query"""
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summaries, query_str)
            for _, community_summary in community_summaries.items()
        ]

        final_answer = self.aggregate_answer(community_answers)
        return final_answer

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM"""
        prompt = (
            f"Given the community summaty: {community_summary}, "
            f"How would you answer the following query ? Query: {query}"
        )

        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user", content="I need an answer based on the above information."
            ),
        ]

        response = self.llm.chat(messages)
        clean_res = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_res

    def aggregate_answer(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""  # Tổng hợp các câu trả lời cộng đồng cá nhân thành một phản hồi cuối cùng, mạch lạc.
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."  # Tạo prompt
        messages = [  # Tạo danh sách messages
            # Tạo tin nhắn hệ thống
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                # Tạo tin nhắn của người dùng
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        # Gửi tin nhắn tới LLM và nhận phản hồi cuối cùng
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(  # Làm sạch phản hồi cuối cùng
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response  # Trả về phản hồi cuối cùng đã làm sạch
