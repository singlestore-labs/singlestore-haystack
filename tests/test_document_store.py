# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret

from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore


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
        return SingleStoreDocumentStore(connection_string=Secret.from_token("root:1@127.0.0.1:3306?local_infile=True"),
                                        embedding_dimension=2, recreate_table=True)

    def test_write_documents(self, document_store: SingleStoreDocumentStore):
        docs = [Document(id="1", content="2", embedding=[1.2, 1.4], meta={"a": 1, "b": {"c": "d"}})]
        assert document_store.write_documents(docs) == 1
        document_store.delete_documents(["1"])
        assert document_store.write_documents(docs) == 1

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)
