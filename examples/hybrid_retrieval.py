# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Before running this example, ensure you have SingleStore installed.
# For a quick setup using Docker:
# docker run \
#     -d --name singlestoredb-dev \
#     -e ROOT_PASSWORD="YOUR SINGLESTORE ROOT PASSWORD" \
#     -p 3306:3306 -p 8080:8080 -p 9000:9000 \
#     ghcr.io/singlestore-labs/singlestoredb-dev:latest

# Install required packages for this example, including singlestore-haystack and other libraries needed
# for Markdown conversion and embeddings generation. Use the following command:
# pip install singlestore-haystack markdown-it-py mdit_plain "sentence-transformers>=2.2.0"

# Download some Markdown files to index.
# git clone https://github.com/anakin87/neural-search-pills

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.singlestore_haystack import (
    SingleStoreBM25Retriever,
    SingleStoreEmbeddingRetriever,
)
from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore

# Set an environment variable `S2_CONN_STR` with the connection string to your SingleStore database.
# e.g., "singlestoredb://USER:PASSWORD@HOST:PORT/DB_NAME"

# Initialize SingleStoreDocumentStore
document_store = SingleStoreDocumentStore(
    table_name="haystack_test",
    embedding_dimension=768,
    recreate_table=True,
)

# Create the indexing Pipeline and index some documents
file_paths = glob.glob("neural-search-pills/pills/*.md")

indexing = Pipeline()
indexing.add_component("converter", MarkdownToDocument())
indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
indexing.add_component("document_embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "splitter")
indexing.connect("splitter", "document_embedder")
indexing.connect("document_embedder", "writer")

indexing.run({"converter": {"sources": file_paths}})

# Create the querying Pipeline and try a query
querying = Pipeline()
querying.add_component("text_embedder", SentenceTransformersTextEmbedder())
querying.add_component("retriever", SingleStoreEmbeddingRetriever(document_store=document_store, top_k=3))
querying.add_component("bm25_retriever", SingleStoreBM25Retriever(document_store=document_store, top_k=3))
querying.add_component(
    "joiner",
    DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=3),
)
querying.connect("text_embedder", "retriever")
querying.connect("bm25_retriever", "joiner")
querying.connect("retriever", "joiner")

query = "cross-encoder"
results = querying.run({"text_embedder": {"text": query}, "bm25_retriever": {"query": query}})

for doc in results["joiner"]["documents"]:
    print(doc)
    print("-" * 10)
