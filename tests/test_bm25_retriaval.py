# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0
import unittest

from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.retrievers.singlestore_haystack import SingleStoreBM25Retriever
from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore
from tests.util import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER


class TestBM25Retrieval(unittest.TestCase):
    def test_embedding_retrieval_dot_product(self):
        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"),
            embedding_dimension=768,
            recreate_table=True,
            fulltext_index_options={
                "analyzer": "standard",
            },
        )

        docs = [
            Document(content="SQL is a standard language for accessing and manipulating databases."),
            Document(content="Explore advanced techniques and functions in SQL for better data manipulation."),
            Document(content="Learn about various optimization techniques to improve database performance."),
            Document(content="Discover how SQL is used in web development to interact with databases."),
            Document(content="An overview of best practices for securing data in SQL databases."),
            Document(content="Using SQL for effective data analysis and reporting."),
            Document(content="Fundamentals of designing a robust and scalable database."),
            Document(content="Tips and techniques for tuning SQL queries for better performance."),
            Document(content="Integrating SQL with Python for data science and automation tasks."),
            Document(content="A comparison of NoSQL and SQL databases and their use cases."),
            Document(content="An introduction to real-time analytics."),
            Document(content="Simple examples of real time analytics."),
            Document(content="Create and maintain effective data dictionaries."),
            Document(content="Designing for scalability."),
        ]

        document_store.write_documents(docs)

        retriever = SingleStoreBM25Retriever(document_store=document_store)
        results = retriever.run(
            query="database",
            top_k=2,
            bm25_function="BM25",
        )["documents"]
        assert len(results) == 2
        assert [results[0].content, results[1].content] == [
            "Fundamentals of designing a robust and scalable database.",
            "Learn about various optimization techniques to improve database performance.",
        ]

        results = retriever.run(
            query="database",
            top_k=2,
            bm25_function="BM25_GLOBAL",
        )["documents"]
        assert len(results) == 2
        assert [results[0].content, results[1].content] == [
            "Fundamentals of designing a robust and scalable database.",
            "Learn about various optimization techniques to improve database performance.",
        ]

        with self.assertRaises(ValueError):
            retriever.run(
                query="database",
                top_k=2,
                bm25_function="BM25_WRONG",
            )

        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"),
            embedding_dimension=768,
            recreate_table=True,
            use_fulltext_index=False,
        )

        with self.assertRaises(ValueError):
            SingleStoreBM25Retriever(document_store=document_store)
