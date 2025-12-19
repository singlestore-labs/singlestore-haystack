import os

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret

from haystack_integrations.components.retrievers.singlestore_haystack import SingleStoreBM25Retriever
from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore

# TODO: rewrite examples
document_store = SingleStoreDocumentStore(
    connection_string=Secret.from_env_var("SINGLESTORE_URL"), database_name="the_one_db", recreate_table=True
)

prompt_template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

pipeline = Pipeline()
pipeline.add_component("retriever", SingleStoreBM25Retriever(document_store=document_store, top_k=5))
pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
pipeline.add_component(
    "llm",
    OpenAIGenerator(
        model="deepseek-ai/deepseek-llm-7b-chat",
        api_key=Secret.from_env_var("AUTH_TOKEN"),
        api_base_url=os.getenv("API_URL"),
    ),
)

pipeline.connect("retriever", "prompt_builder.documents")
pipeline.connect("prompt_builder", "llm")

question = input()
response = pipeline.run({"retriever": {"query": question}, "prompt_builder": {"question": question}})
print(response["llm"]["replies"][0])
