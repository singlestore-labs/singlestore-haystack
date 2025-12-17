# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.singlestore_haystack import \
    SingleStoreDocumentStore


# TODO: improve tests
@pytest.mark.integration
class TestBM25Retrieval:
    def test_embedding_retrieval_dot_product(self):
        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(
                "root:1@127.0.0.1:3306?local_infile=True"),
            embedding_dimension=768, recreate_table=True)

        docs = [
            Document(
                content="SQL is a standard language for accessing and manipulating databases."),
            Document(
                content="Explore advanced techniques and functions in SQL for better data manipulation."),
            Document(
                content="Learn about various optimization techniques to improve database performance."),
            Document(
                content="Discover how SQL is used in web development to interact with databases."),
            Document(
                content="An overview of best practices for securing data in SQL databases."),
            Document(
                content="Using SQL for effective data analysis and reporting."),
            Document(
                content="Fundamentals of designing a robust and scalable database."),
            Document(
                content="Tips and techniques for tuning SQL queries for better performance."),
            Document(
                content="Integrating SQL with Python for data science and automation tasks."),
            Document(
                content="A comparison of NoSQL and SQL databases and their use cases."),
            Document(
                content="An introduction to real-time analytics."),
            Document(
                content="Simple examples of real time analytics."),
            Document(
                content="Create and maintain effective data dictionaries."),
            Document(
                content="Designing for scalability."),
        ]

        document_store.write_documents(docs)
        results = document_store._bm25_retrieval(
            query="database", top_k=2,
        )
        assert len(results) == 2
        assert [results[0].content, results[1].content] == [
            "Fundamentals of designing a robust and scalable database.",
            "Learn about various optimization techniques to improve database performance."]
