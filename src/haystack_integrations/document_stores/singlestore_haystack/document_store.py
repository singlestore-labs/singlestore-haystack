# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import singlestoredb as s2
from typing import Any, Dict, List, Optional, Literal

from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError, \
    MissingDocumentError, DocumentStoreError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from singlestoredb.connection import Cursor, Connection
from singlestoredb.utils.results import Result

from haystack_integrations.document_stores.singlestore_haystack.filter import \
    _convert_filters_to_where_clause_and_params

logger = logging.getLogger(__name__)


def escape_string_literal(s: str):
    return "'" + s.replace("'", "''") + "'"


def escape_identifier(identifier: str):
    return "`" + identifier.replace("`", "``") + "`"


def escape_table(database: str, table: str) -> str:
    return f"{escape_identifier(database)}.{escape_identifier(table)}"


def connection_is_valid(connection: Connection) -> bool:
    """
    Internal method to check if the connection is still valid.
    """

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return True
    except s2.Error as e:
        return False


def escape_tsv(data: str) -> str:
    if data is not None:
        return data.replace("\\", "\\\\").replace("\n", "\\n").replace("\t",
                                                                       "\\t")
    else:
        return "\\N"


def escape_tsv_json(data) -> str:
    if data is not None:
        return escape_tsv(json.dumps(data))
    else:
        return "\\N"


def from_haystack_to_tsv_documents(documents: List[Document]):
    for document in documents:
        id = escape_tsv(document.id)
        content = escape_tsv(document.content)
        meta = escape_tsv_json(document.meta)
        embedding = escape_tsv_json(document.embedding)
        yield "\t".join([id, embedding, content, meta]) + "\n"


def from_s2_to_haystack_documents(res: Result) -> List[Document]:
    documents = []
    for row in res:
        if len(row) > 4:
            document = Document(id=row[0], embedding=row[1], content=row[2],
                                meta=row[3])
        else:
            document = Document(id=row[0], embedding=row[1], content=row[2],
                                meta=row[3], score=row[4])
        documents.append(document)

    return documents


class SingleStoreDocumentStore:

    def __init__(self,
                 connection_string: Secret = Secret.from_env_var("S2_CONN_STR"),
                 database_name: str = "db",
                 table_name: str = "haystack_documents",
                 embedding_dimension: int = 768,
                 recreate_table: bool = False,
                 ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.database_name = database_name
        self.embedding_dimension = embedding_dimension
        self.recreate_table = recreate_table
        self._connection = None
        self._cursor = None
        self._table_initialized = False

    @property
    def cursor(self):
        if self._cursor is None or not connection_is_valid(self._connection):
            self._create_connection()

        return self._cursor

    @property
    def connection(self):
        if self._connection is None or not connection_is_valid(
                self._connection):
            self._create_connection()

        return self._connection

    def _create_connection(self):
        """
        Internal method to create a connection to the SingleStore database.
        """

        # close the connection if it already exists
        if self._connection:
            try:
                self._connection.close()
            except s2.Error as e:
                logger.debug("Failed to close connection: %s", str(e))

        conn_str = self.connection_string.resolve_value() or ""
        connection = s2.connect(conn_str)
        connection.autocommit = True

        self._connection = connection
        self._cursor = self._connection.cursor()

        if not self._table_initialized:
            self._initialize_table()

        return self._connection

    def _initialize_table(self):
        """
        Internal method to initialize the table.
        """
        if self.recreate_table:
            self.delete_table()

        self._create_table_if_not_exists()

        self._table_initialized = True

    def delete_table(self):
        """
        Deletes the table used to store Haystack documents.
        The name of the database (`database_name`) and the name of the table (`table_name`)
        are defined when initializing the `SingleStoreDocumentStore`.
        """
        delete_sql = f"DROP TABLE IF EXISTS {escape_table(self.database_name, self.table_name)}"

        self._execute_sql(
            delete_sql,
            error_msg=f"Could not delete table {self.database_name}.{self.table_name} in SingleStoreDocumentStore",
        )

    def _create_table_if_not_exists(self):
        """
        Creates the table to store Haystack documents if it doesn't exist yet.
        """

        create_sql = f"""CREATE TABLE IF NOT EXISTS {escape_table(self.database_name, self.table_name)} (
        id VARCHAR(128) PRIMARY KEY,
        embedding VECTOR({self.embedding_dimension}, F32),
        content LONGTEXT,
        meta JSON,
        FULLTEXT USING VERSION 2 fi (content)
        )"""  # TODO: add other data
        # TODO: configure index creation

        self._execute_sql(create_sql,
                          error_msg="Could not create table in SingleStoreDocumentStore")

    def _execute_sql(
            self, sql_query: str, params: Optional[tuple] = None,
            error_msg: str = "", cursor: Optional[Cursor] = None
    ):
        """
        Internal method to execute SQL statements and handle exceptions.

        :param sql_query: The SQL query to execute.
        :param params: The parameters to pass to the SQL query.
        :param error_msg: The error message to use if an exception is raised.
        :param cursor: The cursor to use to execute the SQL query. Defaults to self.cursor.
        """

        params = params or ()
        cursor = cursor or self.cursor

        logger.debug("SQL query: %s\nParameters: %s", sql_query, params)

        try:
            cursor.execute(sql_query, params)
        except s2.Error as e:
            self.connection.rollback()
            detailed_error_msg = f"{error_msg}.\nYou can find the SQL query and the parameters in the debug logs."
            raise DocumentStoreError(detailed_error_msg) from e

        return cursor

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """

        sql_count = f"SELECT COUNT(*) FROM {escape_table(self.database_name, self.table_name)}"

        count = self._execute_sql(sql_count,
                                  error_msg="Could not count documents in PgvectorDocumentStore").fetchone()[
            0
        ]
        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> \
            List[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises TypeError: If `filters` is not a dictionary.
        :returns: A list of Documents that match the given filters.
        """
        if filters:
            if not isinstance(filters, dict):
                msg = "Filters must be a dictionary"
                raise TypeError(msg)
            if "operator" not in filters and "conditions" not in filters:
                msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
                raise ValueError(msg)

        sql_filter = f"SELECT * FROM {escape_table(self.database_name, self.table_name)}"

        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters)
            sql_filter += sql_where_clause

        cursor = self._execute_sql(
            sql_filter,
            params,
            error_msg="Could not filter documents from PgvectorDocumentStore.",
            cursor=self.cursor,
        )

        records = cursor.fetchall()
        docs = from_s2_to_haystack_documents(records)
        return docs

    def write_documents(self, documents: List[Document],
                        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """

        if len(documents) > 0:
            if not isinstance(documents[0], Document):
                msg = "param 'documents' must contain a list of objects of type Document"
                raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        policy_map = {
            DuplicatePolicy.NONE: "",
            DuplicatePolicy.FAIL: "",
            DuplicatePolicy.SKIP: "IGNORE DUPLICATE KEY ERRORS",
            DuplicatePolicy.OVERWRITE: "REPLACE"
        }

        sql_load_data = f"LOAD DATA LOCAL INFILE ':stream:' {policy_map.get(policy)} INTO TABLE {escape_table(self.database_name, self.table_name)}(id, embedding, content, meta)"

        try:
            res = self.cursor.execute(sql_load_data,
                                      infile_stream=from_haystack_to_tsv_documents(
                                          documents))
            self.cursor.execute(f"OPTIMIZE TABLE {escape_table(self.database_name, self.table_name)} FLUSH")
            return res
        except s2.IntegrityError as ie:
            self.connection.rollback()
            raise DuplicateDocumentError from ie
        except s2.Error as e:
            self.connection.rollback()
            error_msg = (
                "Could not write documents to SingleStoreDocumentStore. \n"
                "You can find the SQL query and the parameters in the debug logs."
            )
            raise DocumentStoreError(error_msg) from e

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.

        :param object_ids: the object_ids to delete
        """
        if not document_ids:
            return

        document_ids_str = ", ".join(
            escape_string_literal(document_id) for document_id in document_ids)
        delete_sql = f"DELETE FROM {escape_table(self.database_name, self.table_name)} WHERE id IN ({document_ids_str})"

        self._execute_sql(delete_sql,
                          error_msg="Could not delete documents from SingleStoreDocumentStore")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            connection_string=self.connection_string.to_dict(),
            database_name=self.database_name,
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            recreate_table=self.recreate_table,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleStoreDocumentStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"],
                                    ["connection_string"])
        return default_from_dict(cls, data)

    def _embedding_retrieval(
            self,
            query_embedding: List[float],
            *,
            filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
            vector_similarity_function: Optional[
                Literal["dot_product", "euclidean_distance"]] = None,
    ) -> List[Document]:
        """
        Retrieves documents that are most similar to the query embedding using a vector similarity metric.

        This method is not meant to be part of the public interface of
        `SingleStoreDocumentStore` and it should not be called directly.
        `SingleStoreEmbeddingRetriever` uses this method directly and is the public interface for it.

        :returns: List of Documents that are most similar to `query_embedding`
        """

        if not query_embedding:
            msg = "query_embedding must be a non-empty list of floats"
            raise ValueError(msg)
        if len(query_embedding) != self.embedding_dimension:
            msg = (
                f"query_embedding dimension ({len(query_embedding)}) does not match SingleStoreDocumentStore "
                f"embedding dimension ({self.embedding_dimension})."
            )
            raise ValueError(msg)

        # the vector must be a string with this format: "'[3,1,2]'"
        query_embedding_for_singlestore = f"'[{','.join(str(el) for el in query_embedding)}]'"

        if vector_similarity_function == "dot_product":
            score_definition = f"(embedding <*> {query_embedding_for_singlestore})"
        elif vector_similarity_function == "euclidean_distance":
            score_definition = f"(embedding <-> {query_embedding_for_singlestore})"

        sql_select = f"SELECT *, {score_definition} AS score FROM {escape_table(self.database_name, self.table_name)}"

        sql_where_clause = ""
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters)

        # we always want to return the most similar documents first
        # so when using euclidean_distance, the sort order must be ASC
        sort_order = "ASC" if vector_similarity_function == "euclidean_distance" else "DESC"

        sql_sort = f" ORDER BY score {sort_order} LIMIT {top_k}"

        sql_query = sql_select + sql_where_clause + sql_sort

        result = self._execute_sql(
            sql_query,
            params,
            error_msg="Could not retrieve documents from SingleStoreDocumentStore."
        )

        records = result.fetchall()
        docs = from_s2_to_haystack_documents(records)
        return docs

    def _bm25_retrieval(self,
                        query: str,
                        filters: Optional[Dict[str, Any]] = None,
                        top_k: Optional[int] = None):
        """
        Retrieves documents that are most similar to `query`, using the BM25 algorithm.

        This method is not meant to be part of the public interface of
        `SingleStoreDocumentStore` nor called directly.
        `SingleStoreBM25Retriever` uses this method directly and is the public interface for it.

        :param query: Text of the query.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.


        :raises ValueError: If `query` is an empty string.
        :returns: List of Document that are most similar to `query`.
        """

        if query is None:
            msg = "query must not be None"
            raise ValueError(msg)

        # TODO: add several versions
        score_definition = f" BM25({escape_table(self.database_name, self.table_name)}, 'content:{query}')"
        sql_select = f"SELECT *, {score_definition} AS score FROM {escape_table(self.database_name, self.table_name)}"

        sql_where_clause = ""
        params = ()
        if filters:
            sql_where_clause, params = _convert_filters_to_where_clause_and_params(
                filters)

        sql_sort = f" ORDER BY score DESC LIMIT {top_k}"

        sql_query = sql_select + sql_where_clause + sql_sort

        result = self._execute_sql(
            sql_query,
            params,
            error_msg="Could not retrieve documents from SingleStoreDocumentStore."
        )

        records = result.fetchall()
        docs = from_s2_to_haystack_documents(records)
        return docs
