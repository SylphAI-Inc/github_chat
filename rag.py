from typing import Optional, Any, List, Dict
import os
from uuid import uuid4

import adalflow as adal
from adalflow.core.db import LocalDB
from adalflow.core.types import (
    ModelClientType,
    Document,
    Conversation,
    DialogTurn,
    UserQuery,
    AssistantResponse,
)
from adalflow.core.string_parser import JsonParser
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from adalflow.components.data_process import (
    RetrieverOutputToContextStr,
    ToEmbeddings,
    TextSplitter,
)
from adalflow.utils import get_adalflow_default_root_path
from adalflow.core.component import Component
from config import configs


rag_prompt_task_desc = r"""
You are a helpful assistant.

Your task is to answer the query that may or may not come with context information.
When context is provided, you should stick to the context and less on your prior knowledge to answer the query.

Previous conversation history is provided to help you maintain context of the discussion.
Use the conversation history to provide more relevant and contextual answers.

Output JSON format:
{
    "answer": "The answer to the query",
}"""


class Memory(Component):
    def __init__(self, turn_db: LocalDB = None):
        """Initialize the Memory component."""
        super().__init__()
        self.current_conversation = Conversation()
        self.turn_db = turn_db or LocalDB()  # all turns
        self.conver_db = LocalDB()  # a list of conversations

    def call(self) -> str:
        """Returns the current conversation history as a formatted string."""
        if not self.current_conversation.dialog_turns:
            return ""

        formatted_history = []
        for turn in self.current_conversation.dialog_turns.values():
            formatted_history.extend(
                [
                    f"User: {turn.user_query.query_str}",
                    f"Assistant: {turn.assistant_response.response_str}",
                ]
            )
        return "\n".join(formatted_history)

    def add_dialog_turn(self, user_query: str, assistant_response: str):
        """Add a new dialog turn to the current conversation."""
        dialog_turn = DialogTurn(
            id=str(uuid4()),
            user_query=UserQuery(query_str=user_query),
            assistant_response=AssistantResponse(response_str=assistant_response),
        )

        self.current_conversation.append_dialog_turn(dialog_turn)
        self.turn_db.add(
            {"user_query": user_query, "assistant_response": assistant_response}
        )


class RAG(adal.Component):
    def __init__(self, index_path: str = None):
        super().__init__()

        # Initialize memory component
        self.memory = Memory()

        if index_path is None:
            index_path = os.path.join(
                get_adalflow_default_root_path(), "db_adalflow"
            )

        try:
            self.db = LocalDB.load_state(index_path)
            self.transformed_docs = self.db.get_transformed_data("split_and_embed")
        except (FileNotFoundError, KeyError):
            print(f"No existing database found at {index_path}. Initializing new database.")
            self.db = LocalDB("new_db")
            self.transformed_docs = []

        embedder = adal.Embedder(
            model_client=configs["embedder"]["model_client"](),
            model_kwargs=configs["embedder"]["model_kwargs"],
        )
        # map the documents to embeddings
        self.retriever = FAISSRetriever(
            **configs["retriever"],
            embedder=embedder,
            documents=self.transformed_docs,
            document_map_func=lambda doc: doc.vector,
        )
        self.retriever_output_processors = RetrieverOutputToContextStr(deduplicate=True)

        self.generator = adal.Generator(
            prompt_kwargs={
                "task_desc_str": rag_prompt_task_desc,
            },
            model_client=configs["generator"]["model_client"](),
            model_kwargs=configs["generator"]["model_kwargs"],
            output_processors=JsonParser(),
        )

    def generate(self, query: str, context: Optional[str] = None) -> Any:
        if not self.generator:
            raise ValueError("Generator is not set")

        # Add conversation history to context
        full_context = ""
        if context:
            full_context += f"Reference context:\n{context}\n"
        
        # Get conversation history from memory component
        conversation_history = self.memory()
        if conversation_history:
            full_context += f"\nPrevious conversation:\n{conversation_history}"

        prompt_kwargs = {
            "context_str": full_context,
            "input_str": query,
        }
        response = self.generator(prompt_kwargs=prompt_kwargs)
        # Extract just the answer from the GeneratorOutput
        return response.data['answer']

    def call(self, query: str) -> Any:
        retrieved_documents = self.retriever(query)
        # fill in the document
        for i, retriever_output in enumerate(retrieved_documents):
            retrieved_documents[i].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retriever_output.doc_indices
            ]

        context_str = self.retriever_output_processors(retrieved_documents)
        response = self.generate(query, context=context_str)
        
        # Update conversation history in memory
        self.memory.add_dialog_turn(
            user_query=query,
            assistant_response=response
        )

        return response, retrieved_documents


if __name__ == "__main__":
    from adalflow.utils import get_logger

    adal.setup_env()
    rag = RAG()
    print("RAG component initialized. Type your query below or type 'exit' to quit.")

    while True:
        # Get user input
        query = input("Enter your query (or type 'exit' to stop): ")

        # Exit condition
        if query.lower() in ["exit", "quit", "stop"]:
            print("Exiting RAG component. Goodbye!")
            break

        # Process the query
        try:
            response, retrieved_documents = rag(query)
            print(f"\nResponse:\n{response}\n")
            print(f"Retrieved Documents:\n{retrieved_documents}\n")
        except Exception as e:
            print(f"An error occurred while processing the query: {e}")
