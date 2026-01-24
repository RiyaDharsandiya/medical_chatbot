from dotenv import load_dotenv
import os
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,donwload_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

extracted_data=load_pdf_files(data='data/')
filter_data=filter_to_minimal_docs(extracted_data)
text_chunk=text_split(filter_data)

embeddings=donwload_embeddings()

pinecone_api=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api)

index_name="medical-bot"

#store vector in the embedding(pinecone)
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )
    index=pc.Index(index_name)
    
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    index_name=index_name
)
