from flask import Flask, request, jsonify
from flask_cors import CORS
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

app = Flask(__name__)

# Allow requests from frontend (Update with actual frontend URL)
CORS(app, origins=["https://portfoliofrontend-mauve.vercel.app", "http://localhost:3000"])  

# Environment variables
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

# Load or create FAISS vector store
VECTOR_DB_PATH = "faiss_index"
if os.path.exists(VECTOR_DB_PATH):
    vectors = FAISS.load_local(VECTOR_DB_PATH, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)

else:
    vectors = None  # To be initialized later

def construct_system_prompt(chat_history):
    """Constructs system prompt based on chat history."""
    chat_content = "\n".join(
        [f"User: {entry['user']}\nAssistant: {entry['assistant']}"  
         for entry in chat_history]
    )
    return f"""
    Mission Statement
You are a Resume Assistant, dedicated to providing instant and accurate professional details about SK Eswar Sudhan when queried. Your role is to act as a knowledgeable and supportive guide, delivering concise and relevant responses.

Core Guidelines
1. Professional & Concise Communication: Keep responses clear, professional, and to the point. Avoid unnecessary details or over-explanations.
2. Context-Aware Responses: Use chat history to maintain continuity and provide informed answers based on previous interactions.
3. No Glorification: Present information factually without exaggeration or unnecessary praise.
4. No Personal or Sensitive Information: If asked, respond: "Kindly note that this chatbot is for professional purposes only and does not share personal information."
5. Strictly Relevant Data: Provide only details available in the context. Do not make assumptions or add external information.

Interaction Protocol
1. Engage Professionally & Friendly: Respond in an approachable yet professional manner, like a helpful teammate.
2. Context-Driven Responses: Use prior chat history to maintain continuity and avoid repeating information unnecessarily.
3. Direct & Efficient Answers: Keep replies short, relevant, and free of fluff to enhance clarity.
4. No Function Names or Metadata: Do not prepend responses with labels like "Response:" or include function references.
  

    Chat History:
    {chat_content}
    <context>
    {{context}}
    <context>
    
    with that answer:
    Questions:{{input}}
    """


prompt_template = ChatPromptTemplate.from_template(construct_system_prompt([]))

def initialize_vector_embedding():
    """Prepare and persist the FAISS vector store."""
    global vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFDirectoryLoader("./movies")  
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])

    vectors = FAISS.from_documents(final_documents, embeddings)
    vectors.save_local(VECTOR_DB_PATH)  # Save for future use

def save_chat_to_memory(conversation_id, user_input, assistant_response):
    """Save chat history under a unique conversation ID."""
    collection.update_one(
        {"_id": conversation_id},
        {"$push": {"chat_history": {"user": user_input, "assistant": assistant_response}}},
        upsert=True
    )

def retrieve_chat_history(conversation_id):
    """Retrieve chat history based on conversation ID."""
    existing_doc = collection.find_one({"_id": int(conversation_id)})
    return existing_doc["chat_history"] if existing_doc else []

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the vector store."""
    initialize_vector_embedding()
    return jsonify({"message": "Vector Store DB is ready."})

# @app.route('/ask', methods=['POST'])
# def ask():
#     """Handle user questions."""
#     global vectors
#     data = request.json
#     if not data or 'question' not in data or 'conversation_id' not in data:
#         return jsonify({"error": "Missing 'question' or 'conversation_id'."}), 400

#     user_question = data['question']
#     conversation_id = data['conversation_id']

#     # Ensure vector store is initialized
#     if vectors is None:
#         return jsonify({"error": "Vector store not initialized. Call /initialize first."}), 500

#     # Retrieve chat history
#     previous_chats = retrieve_chat_history(conversation_id)
#     system_prompt = construct_system_prompt(previous_chats)
#     prompt_template = ChatPromptTemplate.from_template(system_prompt)

#     # Create the retrieval and document chain
#     document_chain = create_stuff_documents_chain(llm, prompt_template)
#     retriever = vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)

#     # Measure response time
#     start = time.process_time()
#     response = retrieval_chain.invoke({'input': user_question})
#     response_time = time.process_time() - start

#     # Extract response
#     assistant_response = response['answer']

#     # Save conversation history
#     save_chat_to_memory(conversation_id, user_question, assistant_response)

#     # Prepare result
#     result = {
#         "answer": assistant_response,
#         "response_time": response_time,
#         "similar_chunks": [doc.page_content for doc in response.get("context", [])]
#     }

#     return jsonify(result)

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions with locally stored chat history from frontend."""
    global vectors
    data = request.json
    if not data or 'question' not in data or 'conversation_id' not in data or 'chat_history' not in data:
        return jsonify({"error": "Missing 'question', 'conversation_id', or 'chat_history'."}), 400
    
    
    user_question = data['question']
    chat_history = data['chat_history']  # Get last 10 messages from frontend
    
    formatted_chat_history = []
    for i in range(0, len(chat_history) - 1, 2):  # Step of 2
        user_message = chat_history[i].get("user")
        assistant_message = chat_history[i + 1].get("assistant")

        if user_message and assistant_message:
            formatted_chat_history.append({"user": user_message, "assistant": assistant_message})
    # Ensure vector store is initialized
    if vectors is None:
        return jsonify({"error": "Vector store not initialized. Call /initialize first."}), 500

    # Construct system prompt using received chat history
    system_prompt = construct_system_prompt(formatted_chat_history)
    prompt_template = ChatPromptTemplate.from_template(system_prompt)

    # Create the retrieval and document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print(system_prompt)
    # Measure response time
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_question})
    response_time = time.process_time() - start

    # Extract response
    assistant_response = response['answer']
    # print(chat_history)
    # Prepare result
    result = {
        "answer": assistant_response,
        "response_time": response_time,
        # "similar_chunks": [doc.page_content for doc in response.get("context", [])]
    }

    return jsonify(result)


@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Retrieve chat history by conversation ID."""
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({"error": "Missing 'conversation_id'."}), 400

    chat_history = retrieve_chat_history(conversation_id)
    return jsonify({"conversation_id": conversation_id, "chat_history": chat_history})



if __name__ == "__main__":
    app.run()

