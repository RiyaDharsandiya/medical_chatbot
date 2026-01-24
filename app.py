from flask import Flask, render_template, jsonify, request, session
from src.helper import donwload_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.prompt import *
import os

#init flask
app=Flask(__name__)
app.secret_key = "your-secret-key-here"  # Required for session memory

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

#load embedding model and load index
embeddings=donwload_embeddings()
index_name="medical-bot"
docsearch=PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

#chain setup
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  
)

chatModel = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=5,  # Last 5 conversation turns
    return_messages=True
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
        ("placeholder", "{chat_history}")  # Memory placeholder
    ]
)

question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

# Create conversational chain with memory
conversational_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=chatModel,
    retriever=retriever,
    memory=memory,
    combine_docs_chain=question_answer_chain
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route('/get',methods=["GET","POST"])
def chat():
    msg=request.form['msg']
    print(f"User: {msg}")
    response = conversational_rag_chain({"question": msg})
    answer = response["answer"]
    
    print("AI Response:", answer)
    print("Chat History:", memory.chat_memory.messages)
    
    return str(answer)

if __name__=="__main__":
    app.run(host="0.0.0.0" , port=8080, debug=True)
