# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret

from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore
from tests.util import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER


class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self) -> SingleStoreDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        return SingleStoreDocumentStore(
            connection_string=Secret.from_token(f"{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}"),
            embedding_dimension=768,
            recreate_table=True,
        )

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        received.sort(key=lambda doc: doc.id)
        expected.sort(key=lambda doc: doc.id)

        for doc_received, doc_expected in zip(received, expected):
            assert doc_received.id == doc_expected.id
            assert doc_received.content == doc_expected.content
            assert sorted(doc_received.meta.items()) == sorted(doc_expected.meta.items())

            if doc_received.embedding is None and doc_expected.embedding is None:
                continue
            for a, b in zip(doc_received.embedding, doc_expected.embedding):
                assert a - b < 1e-6

    def test_write_documents(self, document_store: SingleStoreDocumentStore):
        docs = [Document(id="1", content="2", embedding=[1.2, 1.4] * 384, meta={"a": 1, "b": {"c": "d"}})]
        assert document_store.write_documents(docs) == 1
        document_store.delete_documents(["1"])
        assert document_store.write_documents(docs) == 1

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_filter(self, document_store: SingleStoreDocumentStore):
        docs = [
            Document(id="1", content="11", embedding=[1.2, 1.4] * 384, meta={"a": 1, "b": {"c": "d"}}),
            Document(id="2", content="22", embedding=[1.2, 1.4] * 384, meta={"a": 2, "b": {"c": "d"}}),
            Document(id="3", content="33", embedding=[1.2, 1.4] * 384, meta={"a": 3, "b": {"c": "d"}}),
            Document(id="4", content="44", embedding=[1.2, 1.4] * 384, meta={"a": 4, "b": {"c": "d"}}),
        ]
        assert document_store.write_documents(docs) == 4

        old_docs = document_store.filter_documents({"field": "meta.a", "operator": "<", "value": 3})
        assert len(old_docs) == 2

    def test_write_documents_duplicate_overwrite(self, document_store: DocumentStore):
        """Test write_documents() overwrites when using DuplicatePolicy.OVERWRITE."""
        # SingleStore treats updated rows in LOAD DATA query as 2 new rows
        # Overridden existing tests to reflect that behavior
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 2
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 2
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])
        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 2
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
