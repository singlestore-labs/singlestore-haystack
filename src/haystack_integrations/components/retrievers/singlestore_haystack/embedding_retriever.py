# SPDX-FileCopyrightText: 2025-present SingleStore, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy, apply_filter_policy

from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore

VALID_VECTOR_SIMILARITY_FUNCTIONS = ["dot_product", "euclidean_distance"]


@component
class SingleStoreEmbeddingRetriever:
    def __init__(
        self,
        *,
        document_store: SingleStoreDocumentStore,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10,
        vector_similarity_function: Optional[Literal["dot_product", "euclidean_distance"]] = "dot_product",
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        if not isinstance(document_store, SingleStoreDocumentStore):
            msg = "document_store must be an instance of SingleStoreDocumentStore"
            raise ValueError(msg)

        if vector_similarity_function and vector_similarity_function not in VALID_VECTOR_SIMILARITY_FUNCTIONS:
            msg = f"vector_similarity_function must be one of {VALID_VECTOR_SIMILARITY_FUNCTIONS}"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.vector_similarity_function = vector_similarity_function
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
            vector_similarity_function=self.vector_similarity_function,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SingleStoreEmbeddingRetriever":
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
        query_embedding: list[float],
        filters: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
        vector_similarity_function: Optional[Literal["dot_product", "euclidean_distance"]] = "dot_product",
        vector_search_options: Optional[dict[str, int]] = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from the `SingleStoreDocumentStore`, based on their embeddings.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See the init method docstring for more
                        details.
        :param top_k: Maximum number of Documents to return.
        :param vector_similarity_function: The similarity function to use when searching for similar embeddings.
        :param vector_search_options: Additional options for vector search, e.g., {"k": 100} to search top 100
                                      candidates before applying top_k and filters.
                                      See SingleStore documentation for more details.
                                      https://docs.singlestore.com/cloud/developer-resources/functional-extensions/tuning-vector-indexes-and-queries/

        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that are similar to `query_embedding`.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        vector_similarity_function = vector_similarity_function or self.vector_similarity_function

        docs = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
            vector_similarity_function=vector_similarity_function,
            vector_search_options=vector_search_options,
        )
        return {"documents": docs}
