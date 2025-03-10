import glob

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.document_stores.singlestore_haystack import SingleStoreDocumentStore

document_store = SingleStoreDocumentStore(
    connection_string=Secret.from_env_var("SINGLESTORE_URL"),
    database_name="the_one_db",
    recreate_table=True
)

pipeline = Pipeline()
pipeline.add_component("converter", TextFileToDocument(encoding='latin-1'))
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("converter", "writer")

pipeline.run({"converter": {"sources": glob.glob("data/*")}})
