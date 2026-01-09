# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.retrievers.singlestore_haystack import SingleStoreEmbeddingRetriever
from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore
from tests.util import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER


class TestEmbeddingRetrieval:
    def test_embedding_retrieval_dot_product(self):
        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"),
            embedding_dimension=768,
            recreate_table=True,
            dot_product_vector_index_options={"index_type": "IVF_PQFS"},
            euclidian_distance_vector_index_options={"index_type": "IVF_PQFS"},
        )

        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        third_best_embedding = [0.7] * 700 + [0.1] * 3 + [0.2] * 65

        docs = [
            Document(content="Most similar document (dot product)", embedding=most_similar_embedding, meta={"a": 1}),
            Document(content="2nd best document (dot product)", embedding=second_best_embedding, meta={"a": 2}),
            Document(content="3rd best document (dot product)", embedding=third_best_embedding, meta={"a": 3}),
        ]

        document_store.write_documents(docs)

        retriever = SingleStoreEmbeddingRetriever(document_store=document_store)
        results = retriever.run(
            query_embedding=query_embedding,
            top_k=1,
            filters={"field": "meta.a", "operator": "!=", "value": 1},
            vector_similarity_function="dot_product",
        )["documents"]
        assert len(results) == 1
        assert results[0].content == "2nd best document (dot product)"

        results = retriever.run(
            query_embedding=query_embedding,
            top_k=2,
            vector_similarity_function="euclidean_distance",
        )["documents"]
        assert len(results) == 2
        assert results[0].content == "3rd best document (dot product)"
        assert results[1].content == "2nd best document (dot product)"

        results = retriever.run(
            query_embedding=query_embedding,
            top_k=2,
            vector_similarity_function="euclidean_distance",
            vector_search_options={"k": 3},
        )["documents"]
        assert len(results) == 2
        assert results[0].content == "3rd best document (dot product)"
        assert results[1].content == "2nd best document (dot product)"

        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"),
            embedding_dimension=768,
            recreate_table=True,
            use_dot_product_vector_index=False,
            use_euclidian_distance_vector_index=False,
        )
        document_store.write_documents(docs)

        retriever = SingleStoreEmbeddingRetriever(document_store=document_store)
        results = retriever.run(
            query_embedding=query_embedding,
            top_k=2,
            vector_similarity_function="euclidean_distance",
        )["documents"]
        assert len(results) == 2
        assert results[0].content == "3rd best document (dot product)"
        assert results[1].content == "2nd best document (dot product)"

        results = retriever.run(
            query_embedding=query_embedding,
            top_k=2,
            vector_similarity_function="euclidean_distance",
            vector_search_options={"k": 3},
        )["documents"]
        assert len(results) == 2
        assert results[0].content == "3rd best document (dot product)"
        assert results[1].content == "2nd best document (dot product)"
