# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.utils import Secret

from haystack_integrations.document_stores.singlestore_haystack import \
    SingleStoreDocumentStore


@pytest.mark.integration
class TestEmbeddingRetrieval:
    def test_embedding_retrieval_dot_product(self):
        document_store = SingleStoreDocumentStore(
            connection_string=Secret.from_token(
                "root:1@127.0.0.1:3306?local_infile=True"),
            embedding_dimension=768, recreate_table=True)

        query_embedding = [0.1] * 768
        most_similar_embedding = [0.8] * 768
        second_best_embedding = [0.8] * 700 + [0.1] * 3 + [0.2] * 65
        third_best_embedding = [0.7] * 700 + [0.1] * 3 + [0.2] * 65

        docs = [
            Document(content="Most similar document (dot product)",
                     embedding=most_similar_embedding, meta={"a": 1}),
            Document(content="2nd best document (dot product)",
                     embedding=second_best_embedding, meta={"a": 2}),
            Document(content="3rd best document (dot product)",
                     embedding=third_best_embedding, meta={"a": 3}),
        ]

        document_store.write_documents(docs)

        results = document_store._embedding_retrieval(
            query_embedding=query_embedding, top_k=1,
            filters={"field": "meta.a", "operator": "!=", "value": 1},
            vector_similarity_function="dot_product"
        )
        assert len(results) == 1
        assert results[0].content == "2nd best document (dot product)"
