from flask import Flask, request, jsonify
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pymongo import MongoClient
import time

# Load environment variables
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')
mongodb_db_name = os.getenv('MONGODB_DB_NAME')
mongodb_collection_name = os.getenv('MONGODB_COLLECTION_NAME')

# Initialize MongoDB client
client = MongoClient(mongodb_uri)
db = client[mongodb_db_name]
collection = db[mongodb_collection_name]

# Initialize the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt
def construct_system_prompt(chat_history):
    chat_content = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in chat_history])
    return f"""
    Answer the questions based on the provided context and chat history only.
    Chat History:
    {chat_content}
    <context>
    {{context}}
    <context>
    Questions:{{input}}
    """

prompt_template = ChatPromptTemplate.from_template(construct_system_prompt([]))

def vector_embedding():
    """Prepare the vector store for document retrieval."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./movies")  # Data ingestion
    docs = loader.load()  # Document loading

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk creation
    final_documents = text_splitter.split_documents(docs[:20])  # Splitting

    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector embeddings
    return vectors

def save_chat_to_memory(user_input, assistant_response):
    """Append conversation to the same MongoDB document."""
    existing_doc = collection.find_one({"_id": "conversation_history"})
    if existing_doc:
        collection.update_one(
            {"_id": "conversation_history"},
            {"$push": {"chat_history": {"user": user_input, "assistant": assistant_response}}}
        )
    else:
        collection.insert_one({
            "_id": "conversation_history",
            "chat_history": [{"user": user_input, "assistant": assistant_response}]
        })

def retrieve_chat_history():
    """Retrieve chat history from MongoDB."""
    existing_doc = collection.find_one({"_id": "conversation_history"})
    return existing_doc["chat_history"] if existing_doc else []

# Flask application
app = Flask(__name__)

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the vector store."""
    global vectors
    vectors = vector_embedding()
    return jsonify({"message": "Vector Store DB is ready."})

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions through a POST API."""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    user_question = data['question']

    # Retrieve chat history from MongoDB
    previous_chats = retrieve_chat_history()
    system_prompt = construct_system_prompt(previous_chats)
    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    # Create the retrieval and document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_question})
    response_time = time.process_time() - start

    # Extract the assistant response
    assistant_response = response['answer']

    # Save chat to memory
    save_chat_to_memory(user_question, assistant_response)

    # Prepare the response
    result = {
        "answer": assistant_response,
        "response_time": response_time,
        "similar_chunks": [doc.page_content for doc in response.get("context", [])]
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port = 8080) 
