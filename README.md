<h1 align="center">singlestore-haystack</h1>

<p align="center">A <a href="https://docs.haystack.deepset.ai/docs/document_store"><i>Haystack</i></a> Document Store for <a href="https://www.singlestore.com/"><i>SingleStore</i></a>.</p>

<p align="center">
  <a href="https://github.com/singlestore-labs/singlestore-haystack/actions?query=workflow%3Atest">
    <img alt="ci" src="https://github.com/singlestore-labs/singlestore-haystack/workflows/test/badge.svg" />
  </a>
<!--- TODO: add documentation link when available
  <a href="link_to_documentation">
    <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat" />
  </a>
-->
  <a href="https://pypi.org/project/singlestore-haystack/">
    <img alt="pypi version" src="https://img.shields.io/pypi/v/singlestore-haystack.svg" />
  </a>
  <a href="https://img.shields.io/pypi/pyversions/singlestore-haystack.svg">
    <img alt="python version" src="https://img.shields.io/pypi/pyversions/singlestore-haystack.svg" />
  </a>
  <a href="https://pypi.org/project/haystack-ai/">
    <img alt="haystack version" src="https://img.shields.io/pypi/v/haystack-ai.svg?label=haystack" />
  </a>
</p>

---

**Table of Contents**

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
    - [Running SingleStore](#running-singlestore)
    - [Writing Documents](#writing-documents)
    - [Index configuration](#index-configuration)
    - [Retrieving documents](#retrieving-documents)
    - [More examples](#more-examples)
- [License](#license)

## Overview

An integration of the [SingleStore](https://www.singlestore.com/) database
with [Haystack](https://docs.haystack.deepset.ai/docs/intro)
by [deepset](https://www.deepset.ai). In
SingleStore [Vector search index](https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/)
is being used for storing document embeddings and dense retrievals and
[Full-Text search index](https://docs.singlestore.com/cloud/developer-resources/functional-extensions/working-with-full-text-search/)
is being used for text-based retrievals.

The library allows using SingleStore as a [DocumentStore](https://docs.haystack.deepset.ai/docs/document-store),
and implements the required [Protocol](https://docs.haystack.deepset.ai/docs/document-store#documentstore-protocol)
methods. You can start working with the implementation by importing it from `singlestore_haystack` package:

```python
from singlestore_haystack import SingleStoreDocumentStore
```

In addition, to the `SingleStoreDocumentStore` the library includes the following haystack components which can be used
in a
pipeline:

- SingleStoreEmbeddingRetriever -
  is a typical [retriever component](https://docs.haystack.deepset.ai/docs/retrievers) which can be used to query
  SingleStore vector index and find semantically related Documents. The component uses
  `SingleStoreDocumentStore` to perform vector similarity search over stored embeddings.

- SingleStoreBM25Retriever -
  is a retriever component that performs sparse retrieval using the BM25 ranking algorithm.
  It leverages SingleStore full-text search capabilities to retrieve Documents based on keyword
  relevance rather than embeddings. The component uses `SingleStoreDocumentStore` to execute
  BM25 queries and is well suited for keyword-based and hybrid search scenarios.

The `singlestore-haystack` library
uses [Python Client](https://pypi.org/project/singlestoredb/) to interact with SingleStore database and hide
all complexities under the hood.

`SingleStoreDocumentStore` will store Documents as rows in SingleStore table.
Embeddings are stored as a [VECTOR](https://docs.singlestore.com/cloud/reference/sql-reference/data-types/vector-type/)
column.

```text
                                         +-----------------------------------+
                                         |       SingleStore Database        |
                                         +-----------------------------------+
                                         |                                   |
                write_documents          |      +----------------------+     |
          +------------------------------+----->|    Haystack table    |     |
          |                              |      +----------------------+     |
+---------+----------------+             |      |  * embedding         |     |
|                          |             |      |  * content           |     |
| SingleStoreDocumentStore |             |      |  * other attributes  |     |
|                          |             |      |                      |     |
+---------+----------------+             |      |  - vector indexes    |     |
          |                              |      |  - fulltext index    |     |
          +------------------------------+----->|                      |     |
                retrieve_documents       |      +----------------------+     |
                                         |                                   |
                                         +-----------------------------------+
```

In the above diagram:

- `Haystack table` is a SingleStore table used by `SingleStoreDocumentStore` to persist Haystack
  Document objects as rows
- `embedding` is also a property of the Document (just shown separately in the diagram for clarity) which is a
  vector of type `VECTOR(n, F32)`.
- `content` is also a property of the Document (just shown separately in the diagram for clarity).
- `vector indexes` are SingleStore vector indexes created on the `embedding` column to enable efficient search for dense
  retrieval.
- `fulltext index` is a SingleStore full-text index created on the `content`
  column to support BM25-based sparse retrieval.
- `write_documents` represents the operation where Documents are inserted in to the table by
  `SingleStoreDocumentStore`.
- `retrieve_documents` represents retrieval operations executed by retrievers, such as
  `SingleStoreEmbeddingRetriever` (vector search) and `SingleStoreBM25Retriever` (full-text search).

`SingleStoreDocumentStore` automatically creates the required vector and full-text indexes if they do not already exist.
When using `SingleStoreEmbeddingRetriever`, Documents must be embedded before they are written to the database.
This can be done using one of the available [Haystack embedders](https://docs.haystack.deepset.ai/docs/embedders).
For example, the
[SentenceTransformersDocumentEmbedder](https://docs.haystack.deepset.ai/docs/sentencetransformersdocumentembedder)
can be used in an indexing pipeline to generate document embeddings prior to persisting them in SingleStore.

## Installation

`singlestore-haystack` can be installed as any other Python library, using pip:

```bash
pip install --upgrade pip # optional
pip install sentence-transformers # required in order to run pipeline examples given below
pip install singlestore-haystack
```

## Usage

### Running SingleStore

You will need to have a running instance of SingleStore database to use components from the package.
The simplest way to start a database locally will be with a Docker container:

```bash
docker run \
    -d --name singlestoredb-dev \
    -e ROOT_PASSWORD="YOUR SINGLESTORE ROOT PASSWORD" \
    -p 3306:3306 -p 8080:8080 -p 9000:9000 \
    ghcr.io/singlestore-labs/singlestoredb-dev:latest
```

### Writing documents

Once you have the package installed and the database running, you can start using `SingleStoreDocumentStore` as any
other document stores that support embeddings.

export `S2_CONN_STR` environment variable with your connection string to avoid hardcoding credentials in the code:

```bash
export S2_CONN_STR="singlestoredb://USER:PASSWORD@HOST:PORT"
```

```python
from singlestore_haystack import SingleStoreDocumentStore

document_store = SingleStoreDocumentStore(
    database_name="haystack_db",  # The name of the database in SingleStore
    table_name="haystack_documents",  # The name of the table to store Documents
    embedding_dimension=384  # The dimension of the embeddings being stored
)
```

Assuming there is a list of documents available and a running SingleStore database you can write those in SingleStore,
e.g.:

```python
from haystack import Document

documents = [Document(content="My name is Morgan and I live in Paris.")]

document_store.write_documents(documents)
```

If you intend to obtain embeddings before writing documents, use the following code:

```python
from haystack import Document

# import one of the available document embedders
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

documents = [Document(content="My name is Morgan and I live in Paris.")]

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_embedder.warm_up()  # will download the model during the first run
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"))
```

Make sure embedding model produces vectors of same size as it has been set on `SingleStoreDocumentStore`, e.g. setting
`embedding_dimension=384` would comply with the "sentence-transformers/all-MiniLM-L6-v2" model.

> **Note**
> Most of the time you will be using [Haystack Pipelines](https://docs.haystack.deepset.ai/docs/pipelines) to build both
> indexing and querying RAG scenarios.

It is important to understand how haystack Documents are stored in SingleStore after you call `write_documents`.

```python
from random import random
from haystack import Document

sample_embedding = [random() for _ in range(384)]  # using fake/random embedding for brevity here to simplify example
document = Document(
    content="My name is Morgan and I live in Paris.", embedding=sample_embedding, meta={"num_of_years": 3}
)
print(document.to_dict())
```

The above code converts a Document to a dictionary and will render the following output:

```bash
>>> output:
{
    "id": "945a32e1d4532f8506fc812d5a77b0812cafdd289d6c1af468ee0626129d6ab4",
    "content": "My name is Morgan and I live in Paris.",
    "blob": None,
    "score": None,
    "embedding": [0.814127326,0.327150941,0.166730702, ...], # vector of size 384
    "sparse_embedding": None,
    "num_of_years": 3,
}
```

The data from the dictionary will be used to add a row in SingleStore after you write the document with
`document_store.write_documents([document])`.
Below is a representation of the row in SingleStore:

```bash
singlestore> SET vector_type_project_format = JSON;        
Query OK, 0 rows affected (0.00 sec)

singlestore> select * from haystack_db.haystack_documents\G
*************************** 1. row ***************************
            id: 945a32e1d4532f8506fc812d5a77b0812cafdd289d6c1af468ee0626129d6ab4
     embedding: [0.814127326,0.327150941,0.166730702,... ]  # vector of size 384
       content: My name is Morgan and I live in Paris.
     blob_data: NULL
     blob_meta: NULL
blob_mime_type: NULL
          meta: {"num_of_years":3}
1 row in set (0.06 sec)

```

With Haystack, you can use the [DocumentWriter](https://docs.haystack.deepset.ai/docs/documentwriter) component to
write Documents into a Document Store. In the example below we construct a pipeline to write documents to SingleStore
using
`SingleStoreDocumentStore`:

```python
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter

from singlestore_haystack import SingleStoreDocumentStore

documents = [Document(content="This is document 1"), Document(content="This is document 2")]

document_store = SingleStoreDocumentStore(
    database_name="haystack_db",  # The name of the database in SingleStore
    table_name="haystack_documents",  # The name of the table to store Documents
    embedding_dimension=384,  # The dimension of the embeddings being stored
    recreate_table=True,  # recreate the table if it already exists
)
embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_writer = DocumentWriter(document_store=document_store)

pipeline = Pipeline()
pipeline.add_component(instance=embedder, name="embedder")
pipeline.add_component(instance=document_writer, name="writer")

pipeline.connect("embedder", "writer")
print(pipeline.run({"embedder": {"documents": documents}}))
```

```bash
>>> output:
`{'writer': {'documents_written': 2}}`
```

### Index configuration

`SingleStoreDocumentStore` allows you to control which indexes are created and used in the database.
Depending on your retrieval strategy, you can enable or disable specific index types and customize their creation
options.

#### Dot product vector index

- `use_dot_product_vector_index`  
  Whether to create and use a vector index optimized for **dot product similarity**.
  Dot product similarity is typically used with **normalized embeddings**.

- `dot_product_vector_index_options`  
  Optional dictionary containing additional options passed to the dot product vector index creation.
  These options are forwarded directly to SingleStore.  
  See the SingleStore documentation for supported options:  
  https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/#index-options

#### Euclidean (L2) distance vector index

- `use_euclidian_distance_vector_index`  
  Whether to create and use a vector index optimized for **Euclidean (L2) distance**.
  This metric is commonly used when embeddings are **not normalized**.

- `euclidian_distance_vector_index_options`  
  Optional dictionary containing additional options passed to the Euclidean distance vector index
  creation. These options are forwarded directly to SingleStore.  
  See the SingleStore documentation for supported options:  
  https://docs.singlestore.com/cloud/reference/sql-reference/vector-functions/vector-indexing/#index-options

#### Full-text index

- `use_fulltext_index`  
  Whether to create and use a **full-text index** for keyword-based retrieval.

- `fulltext_index_options`  
  Optional dictionary containing additional options passed to the full-text index creation, such as
  custom analyzers or tokenization settings.  
  See the SingleStore documentation for details:  
  https://docs.singlestore.com/db/v9.0/developer-resources/functional-extensions/full-text-version-2-custom-analyzers

The full-text index is required for keyword-based retrieval using `SingleStoreBM25Retriever`.

#### Hybrid retrieval

Both vector and full-text indexes can be enabled at the same time to support **hybrid retrieval**
scenarios, where dense (semantic) and sparse (keyword-based) search techniques are combined within
the same Haystack pipeline.

### Retrieving documents

`SingleStoreEmbeddingRetriever` component can be used to retrieve documents from SingleStore by using vector index.
Below is a pipeline which finds documents using vector index as well
as [metadata filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering):

```python
from typing import List

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

from singlestore_haystack import SingleStoreDocumentStore, SingleStoreEmbeddingRetriever

document_store = SingleStoreDocumentStore(
    database_name="haystack_db",  # The name of the database in SingleStore
    table_name="haystack_documents",  # The name of the table to store Documents
    embedding_dimension=384,  # The dimension of the embeddings being stored
    recreate_table=True,
)

documents = [
    Document(content="My name is Morgan and I live in Paris.", meta={"num_of_years": 3}),
    Document(content="I am Susan and I live in Berlin.", meta={"num_of_years": 7}),
]

# The same model is used for both query and Document embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"

document_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"))

print("Number of documents written: ", document_store.count_documents())

pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model_name))
pipeline.add_component("retriever", SingleStoreEmbeddingRetriever(document_store=document_store))
pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = pipeline.run(
    data={
        "text_embedder": {"text": "What cities do people live in?"},
        "retriever": {
            "top_k": 5,
            "filters": {"field": "meta.num_of_years", "operator": "==", "value": 3},
        },
    }
)

documents: List[Document] = result["retriever"]["documents"]
print(documents)
```

```bash
>>> output:
[Document(id=4014455c3be5d88151ba12d734a16754d7af75c691dfc3a5f364f81772471bd2, content: 'My name is Morgan and I live in Paris.', meta: {'num_of_years': 3}, score: 0.33934953808784485, embedding: vector of size 384)]
```

### More examples

You can find more examples in the
implementation [repository](https://github.com/singlestore-labs/singlestore-haystack/tree/main/examples):

- [embedding_retrieval.py](https://github.com/singlestore-labs/singlestore-haystack/tree/main/examples/embedding_retrieval.py) -
- [hybrid_retrieval.py](https://github.com/singlestore-labs/singlestore-haystack/tree/main/examples/hybrid_retrieval.py)

## License

`singlestore-haystack` is distributed under the terms of
the [Apache-2.0](https://github.com/singlestore-labs/singlestore-haystack/blob/main/LICENSE) license.
