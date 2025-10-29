from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pymongo import MongoClient
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Allow requests from frontend
CORS(app, origins=["https://portfoliofrontend-mauve.vercel.app", "http://localhost:3000", "https://skeswarsudhan.vercel.app"])  

# Environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
mongodb_uri = os.getenv('MONGODB_URI')
mongodb_db_name = os.getenv('MONGODB_DB_NAME')
mongodb_collection_name = os.getenv('MONGODB_COLLECTION_NAME')

# Initialize MongoDB client (optional - for chat history storage)
if mongodb_uri:
    client = MongoClient(mongodb_uri)
    db = client[mongodb_db_name]
    collection = db[mongodb_collection_name]
else:
    collection = None

# Initialize the LLM
llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=4000
)

# Resume content embedded directly
RESUME_CONTEXT = """
SK Eswar Sudhan
+91-7708450577 | Mail: skeswarsudhan@gmail.com
LinkedIn: https://www.linkedin.com/in/skeswarsudhan/
Github: https://github.com/skeswarsudhan/

PROFESSIONAL SUMMARY
A detail-oriented and innovative Artificial Intelligence Engineer with a strong foundation in Computer Science from Amrita Vishwa Vidyapeetham. Experienced in the end-to-end development of AI-powered applications, from backend architecture and microservices to multimodal chatbot systems and data-driven model validation. Proven ability to enhance system efficiency, optimize code quality, and deliver impactful solutions for enterprise-level clients.

EDUCATION
• Amrita Vishwa Vidyapeetham, Coimbatore (2020 - 2024)
  - B.Tech in Computer Science and Engineering (Artificial Intelligence) | CGPA: 7.98/10
• Maharishi International Residential School (CBSE), Chennai (2018 - 2020)
  - All India Senior School Certificate Examination | Percentage: 93.4%

TECHNICAL SKILLS
• Programming Languages: Python, Java, MATLAB, SQL
• AI & Machine Learning: TensorFlow, PyTorch, Scikit-learn, Pandas, LangChain, Hugging Face, OpenCV
• Web Development & Frameworks: Django REST Framework, Flask, FastAPI, React.js, HTML, CSS, JavaScript, Streamlit
• Databases: MySQL, MongoDB
• Developer Tools & Platforms: Git, Vercel, Docker, Office 365
• Networking & IoT: IRC Protocol

PROFESSIONAL EXPERIENCE

Python Developer (Development & AI Code Quality) | Turing (Aug 2024 - Present)
 • Developed and optimized the backend architecture for an AI-powered hiring platform using Django REST Framework, improving the scalability of candidate screening workflows.
 • Built and deployed Python microservices for resume parsing and job to candidate matching, directly reducing recruiter effort by over 60%.
 • Debugged and resolved critical bugs and scalability issues in LLM-generated code across FastAPI for enterprise clients including Apple, ByteDance, and Amazon.

Software Engineer Intern | FocusR (Feb 2024 - May 2024)
• Developed the backend for a campus placement automation web application using Django REST Framework, enabling seamless interaction between students and campus placement staff.
• Designed and implemented role-based workflows where staff could manage visiting companies, define eligibility criteria, and monitor student applications, while students could apply to qualifying job postings.
• Automated the export of shortlisted candidate data as Excel sheet, streamlining placement operations and reducing the manual effort of placement coordinators by up to 70%.

PROJECTS

EmoBot: Multimodal Emotion-Aware Chatbot
• Engineered a computer vision pipeline using OpenCV to capture video streams and perform real-time face detection with Haar Cascades.
• Trained a custom ResNet CNN with TensorFlow on a labeled dataset of facial expressions to accurately classify emotions.
• Integrated the emotion detection module with a Cohere Large Language Model via LangChain, enabling the chatbot to adapt based on the user's detected emotional state.
• Deployed the application using Streamlit for a user-friendly interface.

Advanced Image Dehazing Pipeline
• Implemented a state-of-the-art image dehazing algorithm based on "Efficient Image Dehazing with Boundary Constraint and Contextual Regularization" (Meng et al., ICCV 2013).
• Developed a multi-stage pipeline including Airlight Estimation, Transmission Map Estimation with contextual regularization, and haze removal.
• Created a Synthetic Hazing script and evaluation module using PSNR and SSIM metrics.

CCTV Motion Detection with IRC Alerting System
• Built a real-time motion detection system using OpenCV and frame-differencing algorithm.
• Developed a custom IRC client to communicate with a Raspberry Pi-hosted IRC server.
• Integrated motion detection with IRC alerting, triggering automated messages upon detection.

AI-Powered Satellite Image Analysis and Chatbot for Disaster Management
• Architected a RESTful backend using Flask to manage image uploads and orchestrate AI model inferences.
• Integrated Hugging Face models via LangChain for image captioning and Visual Question Answering (VQA).
• Developed a dynamic frontend using React.js for seamless user interaction.

Covid-19 Forecasting using DMD and Comparative Analysis
• Implemented Dynamic Mode Decomposition (DMD) in MATLAB and Python to forecast COVID-19 cases.
• Benchmarked DMD against SARIMAX, ARIMA, and ARMA models.
• Achieved RMSE of 969 cases, outperforming traditional models by 30%.

Portfolio Website with Integrated RAG Chatbot
• Developed a modern frontend with React.js and backend with Flask.
• Integrated a Retrieval-Augmented Generation (RAG) pipeline using Ollama and LangChain.
• Deployed on Vercel, resulting in 70% increase in user interaction time.

JPEG-like Image Compression using Discrete Cosine Transform (DCT)
• Implemented JPEG compression algorithm in MATLAB using YCbCr color space conversion.
• Applied DCT to 8x8 pixel blocks and performed quantization using standard JPEG matrix.
• Built complete compression/decompression pipeline demonstrating understanding of lossy compression.

CERTIFICATIONS & ACHIEVEMENTS
• Qualified GATE (DA) 2024: All India Rank (AIR) 5804
• Amazon Proof of Concept (PoC) Contributor: Developed and delivered PoC project for Amazon
• AWS Cloud Essentials Badge: Foundational knowledge of AWS Cloud services
"""

def construct_system_prompt(chat_history):
    """Constructs system prompt with embedded resume context."""
    chat_content = "\n".join(
        [f"User: {entry['user']}\nAssistant: {entry['assistant']}"  
         for entry in chat_history[-5:]]
    )
    
    return f"""You are a Resume Assistant for SK Eswar Sudhan, dedicated to providing instant and accurate professional details when queried. Your role is to act as a knowledgeable and supportive guide, delivering concise and relevant responses.

CORE GUIDELINES
1. Professional & Concise Communication: Keep responses clear, professional, and to the point. Avoid unnecessary details or over-explanations.
2. Context-Aware Responses: Use chat history to maintain continuity and provide informed answers based on previous interactions.
3. No Glorification: Present information factually without exaggeration or unnecessary praise.
4. No Personal or Sensitive Information: If asked for personal details beyond professional information, respond: "Kindly note that this chatbot is for professional purposes only and does not share personal information."
5. Strictly Relevant Data: Provide only details available in the resume context below. Do not make assumptions or add external information.

INTERACTION PROTOCOL
1. Engage Professionally & Friendly: Respond in an approachable yet professional manner, like a helpful teammate.
2. Context-Driven Responses: Use prior chat history to maintain continuity and avoid repeating information unnecessarily.
3. Direct & Efficient Answers: Keep replies fully satisfying the question, relevant, and free of fluff to enhance clarity.
4. No Function Names or Metadata: Do not prepend responses with labels like "Response:" or include function references.

RECENT CHAT HISTORY:
{chat_content}

RESUME INFORMATION:
{RESUME_CONTEXT}

USER QUESTION: {"{input}"}

Provide a comprehensive response using only the information from the resume above:"""

def save_chat_to_memory(conversation_id, user_input, assistant_response):
    """Save chat history under a unique conversation ID."""
    if collection:
        collection.update_one(
            {"_id": conversation_id},
            {"$push": {"chat_history": {"user": user_input, "assistant": assistant_response}}},
            upsert=True
        )

def retrieve_chat_history(conversation_id):
    """Retrieve chat history based on conversation ID."""
    if collection:
        existing_doc = collection.find_one({"_id": int(conversation_id)})
        return existing_doc["chat_history"] if existing_doc else []
    return []

@app.route('/ask', methods=['POST'])
def ask():
    """Handle user questions using direct context (no RAG)."""
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' field."}), 400
    
    user_question = data['question']
    chat_history = data.get('chat_history', [])
    
    # Format chat history from frontend
    formatted_chat_history = []
    for i in range(0, len(chat_history) - 1, 2):
        user_message = chat_history[i].get("user")
        assistant_message = chat_history[i + 1].get("assistant")
        if user_message and assistant_message:
            formatted_chat_history.append({"user": user_message, "assistant": assistant_message})
    
    try:
        # Construct prompt with resume context
        system_prompt = construct_system_prompt(formatted_chat_history)
        
        # Create prompt template and chain
        prompt_template = ChatPromptTemplate.from_template(system_prompt)
        output_parser = StrOutputParser()
        chain = prompt_template | llm | output_parser
        
        # Measure response time
        start = time.process_time()
        assistant_response = chain.invoke({"input": user_question})
        response_time = time.process_time() - start
        
        result = {
            "answer": assistant_response,
            "response_time": response_time
        }

        return jsonify(result)
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    """Retrieve chat history by conversation ID."""
    conversation_id = request.args.get('conversation_id')
    if not conversation_id:
        return jsonify({"error": "Missing 'conversation_id'."}), 400

    try:
        chat_history = retrieve_chat_history(conversation_id)
        return jsonify({"conversation_id": conversation_id, "chat_history": chat_history})
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve chat history: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "llm_ready": True,
        "mode": "direct_context"
    })

if __name__ == "__main__":
    app.run()
