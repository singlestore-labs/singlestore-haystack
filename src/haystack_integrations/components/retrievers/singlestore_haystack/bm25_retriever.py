# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy, apply_filter_policy

from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore


@component
class SingleStoreBM25Retriever:
    def __init__(
        self,
        *,
        document_store: SingleStoreDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        if not isinstance(document_store, SingleStoreDocumentStore):
            msg = "document_store must be an instance of SingleStoreDocumentStore"
            raise ValueError(msg)
        if document_store.use_fulltext_index is False:
            msg = "SingleStoreBM25Retriever requires 'use_fulltext_index' to be True in the SingleStoreDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SingleStoreBM25Retriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = SingleStoreDocumentStore.from_dict(doc_store_params)
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        bm25_function: Literal["BM25", "BM25_GLOBAL"] = "BM25",
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `SingleStoreDocumentStore`, based on their embeddings.

        :param query: String to search in the `Document`s text.
                      SingleStore supports the following operators in the query: https://docs.singlestore.com/cloud/reference/sql-reference/full-text-search-functions/bm-25/#operators
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See the init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return.
        :param bm25_function: The BM25 function to use. Currently, only "BM25" and "BM25_GLOBAL" are supported.
        The BM25 function is more efficient, but potentially less accurate than the BM25_GLOBAL function.
        However, with careful data distribution, BM25 can provide accurate scores, comparable to those of BM25_GLOBAL.

        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that match the query.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k

        docs = self.document_store._bm25_retrieval(
            query=query, filters=filters, top_k=top_k, bm25_function=bm25_function
        )
        return {"documents": docs}
