from flask import Flask, render_template, jsonify, request
from src.helper import donwload_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


#init flask
app=Flask(__name__)

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

#chain
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3}  
)
chatModel = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free",  # Free model from OpenRouter
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ]
)


#creating chain
question_answer_chain=create_stuff_documents_chain(chatModel,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

#route
@app.route("/")
def index():
    return render_template('chat.html')

#when user clicks on send button
@app.route('/get',methods=["GET","POST"])
def chat():
    msg = request.form['msg']
    print(msg)

    response = rag_chain.invoke({
        "input": msg,
        "chat_history": memory.chat_memory.messages
    })

    memory.chat_memory.add_user_message(msg)
    memory.chat_memory.add_ai_message(response["answer"])

    print("Response:", response["answer"])
    return str(response["answer"])


#execute the app
if __name__=="__main__":
    app.run(host="0.0.0.0" , port=8080, debug=True)


