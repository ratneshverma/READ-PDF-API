import json
from langserve import CustomUserType
from pydantic import Field
import base64
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from app.Utility.VectorDBUtil import RemoveVectors


embeddings = OllamaEmbeddings(model='nomic-embed-text')

index = faiss.IndexFlatL2(len(embeddings.embed_query("galaxbiotech-vectorDB")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


class KnowledgeRequest(CustomUserType):
    site: str
    file: str = Field(..., extra={"widget": {"type": "base64file"}})

def GrabFieldValue(field):
    fieldValue = ""
    match field["type"]:
        case "date":
            fieldValue = field["value"]["day"] + "-" + field["value"]["month"] + "-" + str(field["value"]["year"])
        case "number":
            fieldValue = str(field["value"]["newNumberValue"]) + " " + field["value"]["unit"]
        case "radio":
            fieldValue = field["value"]["value"]
        case "text":
            fieldValue = field["value"]["newTextValue"] + " " + field["value"]["unit"]
        case "time":
            fieldValue = field["value"]["hour"] + " " + field["value"]["minute"] + " " + field["value"]["second"]
        case "textarea":
            fieldValue = field["value"]["value"]
        case "lineScale":
            fieldValue = field["value"]["value"] 
        case "checkbox":
            fieldValue = ','.join(field["value"]["value"])
        case default:
            return ""
    return fieldValue.strip()    

def knowledge_base(request: KnowledgeRequest) -> str:
    # new_vector_store = FAISS.load_local(
    # "galaxbiotech-vectorDB", embeddings, allow_dangerous_deserialization=True)
    #d=new_vector_store.docstore._dict
    # results = new_vector_store.similarity_search(
    # "Populate all fields from demographic which have value2",
    # k=2,
    # filter={"source": {"$eq": "Site1-02-01-01-V0_SCR-Demographic"}})
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")


    content = base64.b64decode(request.file.encode("utf-8"))
    Packets = json.loads(content)
    documents = []
    for packet in Packets:
        fieldList = []
        sitename = request.site
        formname= packet['formName']
        subjectname = packet['subjectName']
        vistname = packet['visitName']
        metaindex = f"{sitename}-{subjectname}-{vistname}-{formname}"
        for field in packet['forms']:
            fieldList.append("'{}':'{}'".format(field['label'], GrabFieldValue(field)))
        if len(fieldList) > 0:
            fieldList.append("'{}':'{}'".format("site",sitename))
            fieldList.append("'{}':'{}'".format("subject",subjectname))
            fieldList.append("'{}':'{}'".format("visit",vistname))
            fieldList.append("'{}':'{}'".format("form",formname))
            jsoncontent = "{" + ','.join(fieldList) + "}"
            document_1 = Document(
            page_content=jsoncontent,
            metadata={"source": metaindex})
            documents.append(document_1)
    
    
    #delete existing
    reqcontent = f"please find the values for site {sitename}, subject {subjectname} and visit {vistname} for {formname}"
    new_vector_store = FAISS.load_local(
    "galaxbiotech-vectorDB", embeddings, allow_dangerous_deserialization=True)
    content_raw = new_vector_store.similarity_search(   
                    reqcontent,
                    k=1,
                    filter={"source": {"$eq": metaindex}})
    if len(content_raw) > 0:
        idList = []
        idList.append(content_raw[0].id)
        RemoveVectors(new_vector_store, idList)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]
    new_vector_store.add_documents(documents=documents, ids=uuids)
    new_vector_store.save_local("galaxbiotech-vectorDB")
    return "Successfully vectorize and stored in knowledge base."
