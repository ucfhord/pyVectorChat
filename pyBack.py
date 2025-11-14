import os
import time
from flask import Flask, request, jsonify
from pymongo import MongoClient
#from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pymongo.server_api import ServerApi


print("✅ Libraries imported!")

# ------------------------------------------------------------------

# Step 3: Load Secrets and Initialize Connections
# --- Load Secrets from Colab Secrets Manager (🔑 icon) ---
MONGO_URI = 'mongodb+srv://youcefhord_db_user:bKo3Q1IlAFPUJzXG@chatbot.tcihuxs.mongodb.net/?appName=chatbot'
GEMINI_API_KEY = 'AIzaSyDwbIYwUL0D6-5Vx5IFfTk-4AZbr8XI0IE'

# Check if secrets are loaded
#if not MONGO_URI or not GEMINI_API_KEY:
 #   raise ValueError("Please configure MONGO_URI and GEMINI_API_KEY in Colab Secrets (click the 🔑 icon).")

print("✅ Secrets configured successfully!")

# --- Initialize Connections and Models ---
try:
    mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = mongo_client.chatbot
    collection = db.documents
    # Test connection
    mongo_client.server_info()
    print(f"✅ Connected to MongoDB Atlas! Collection: {collection.name}")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    # Stop execution if DB connection fails
    raise

# --- LangChain Models ---
#embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GEMINI_API_KEY, temperature=0.3) # Changed model to gemini-1.5-flash-latest
print("✅ Initialized embedding model and LLM.")

# ------------------------------------------------------------------

# Step 4: Create the MongoDB Atlas Vector Search Index
# This step is CRUCIAL for the $vectorSearch pipeline to work.
# We will create an index named "default", which the query code expects.

from pymongo.errors import OperationFailure

index_name = "default"
# Corrected index definition for Atlas Vector Search
index_definition = {
    "name": index_name,
    "definition": {
        "mappings": {
            "dynamic": True, # Allows other fields to be indexed automatically
            "fields": [
                {
                    "type": "knnVector",
                    "path": "embedding",
                    "numDimensions": 384,  # Dimensions of all-MiniLM-L6-v2
                    "similarity": "cosine"
                }
            ]
        }
    }
}


# try:
#     # Check if index already exists
#     index_exists = False
#     for index in collection.list_search_indexes():
#         if index['name'] == index_name:
#             index_exists = True
#             break

#     if not index_exists:
#         print(f"Creating vector search index '{index_name}'...")
#         # Note: You need Atlas Admin role to create an index
#         collection.create_search_index(index_definition)
#         print(f"✅ Vector search index '{index_name}' created successfully.")
#     else:
#         print(f"✅ Vector search index '{index_name}' already exists.")

# except OperationFailure as e:
#     print(f"❌ Error creating vector index. Please ensure you have Admin privileges on Atlas.")
#     print(f"Error details: {e.details}")
#     raise

print(f"✅ Skipping vector search index creation step as it likely already exists.")


# ------------------------------------------------------------------

# Step 5: Ingest and Store Document Chunks
# Optional: Clean up the collection before starting
#collection.delete_many({})
#print("🧹 Cleared existing documents from the collection.")

# Sample document about Mars
sample_text = """
Mars, often called the 'Red Planet', is the fourth planet from the Sun and the second-smallest planet in the Solar System.
Named after the Roman god of war, it has a thin atmosphere composed primarily of carbon dioxide.
The surface of Mars is dusty, cold, and desert-like, featuring canyons, volcanoes, and polar ice caps.
One of the most notable features is Olympus Mons, the largest volcano and second-highest known mountain in the Solar System.
Valles Marineris is a system of canyons that runs along the Martian equator, stretching for 4,000 km.
Evidence suggests that Mars was once much wetter and warmer, possibly harboring life.
Missions like the Perseverance rover are currently searching for signs of ancient microbial life.
The rover is collecting rock and soil samples for a possible return to Earth.
Future missions aim to explore the possibility of human colonization.
"""

# 1. Chunk the document
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#chunks = text_splitter.split_text(sample_text)

#print(f"📄 Document split into {len(chunks)} chunks.")

# 2. Generate embeddings for each chunk
#embeddings = embedding_model.embed_documents(chunks)
#print(f"✨ Generated {len(embeddings)} embeddings.")

# 3. Prepare documents for MongoDB insertion
#documents_to_insert = []
#for i, chunk in enumerate(chunks):
#    documents_to_insert.append({
#        "source": "Mars Fact Sheet",
#        "text": chunk,
#        "embedding": embeddings[i] # The vector embedding
#    })

# 4. Insert into Atlas
#collection.insert_many(documents_to_insert)
#print(f"✅ Successfully inserted {len(documents_to_insert)} documents into MongoDB Atlas.")

# ------------------------------------------------------------------
def get_or_create_session_state(session_id: str):
    """
    Fetches the session state from MongoDB, or creates a new one.
    """
    state = conv_collection.find_one({"session_id": session_id})
    if not state:
        state = {
            "session_id": session_id, "last_updated": time.time(),
            "token_counts": {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0},
            "history": [] # Stored as serialized dicts
        }
    
    # Convert dict history back into LangChain Message objects
    message_objects = [loads(msg) for msg in state["history"]]
    return state, message_objects
# Step 6: Query the Vector DB and Perform RAG
def ask_question_with_rag(question: str):
    """
    Performs a RAG query:
    1. Embeds the question.
    2. Performs vector search in Atlas.
    3. Constructs a prompt with the retrieved context.
    4. Calls the LLM to generate an answer.
    """

    # 1. Embed the user's question
    query_embedding = embedding_model.embed_query(question)

    # 2. Define the vector search pipeline
    vector_search_pipeline = [
        {
            "$vectorSearch": {
                "index": "default",  # The name of your vector search index
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": 3  # Retrieve top 3 most relevant chunks
            }
        },
        {"$project": {"_id": 0, "text": 1, "source": 1, "score": {"$meta": "vectorSearchScore"}}}
    ]

    # 3. Execute the search
    results = list(collection.aggregate(vector_search_pipeline))
    print(f"✅ Found!!! {len(results)} results.")
    if not results:
        return "I'm sorry, I couldn't find any relevant information in my knowledge base.", []

    # 4. Format the context and sources
    context = "\n\n---\n\n".join([doc['text'] for doc in results])
    sources = list(set([doc['source'] for doc in results]))
    print(f"📚 found Sources: {sources}")

    #
    # 5. Construct the prompt
    prompt = f"""
    You are an expert AI assistant. Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, say that you don't have enough information on the special context of the user and mention that you gave a general answer.

    Context:
    {context}

    Question:
    {question}
    """

    # 6. Generate the answer
    response = llm.invoke(prompt)

    print("\n--- Retrieved Context ---")
    print(context)
    print("\n--- AI Response ---")
    return response.content, sources

# --- Let's ask a question! ---
#question = "What is Olympus Mons?"
question = "How does Mars compares to the Moon?"
answer, sources = ask_question_with_rag(question)

print(f"❓ Question: {question}")
print(f"✅ Answer: {answer}")
print(f"📚 Sources: {sources}")

# ------------------------------------------------------------------

# Step 7: Adding Conversational Memory
# In-memory store for session memories
session_memories = {}

def ask_conversational_rag(question: str, session_id: str):
    """
    A conversational RAG function that uses session memory.
    """
    # 1. Get or create memory for the session
    #if session_id not in session_memories:
    #    session_memories[session_id] = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="history")
    #memory = session_memories[session_id]
    state, chat_history_messages = get_or_create_session_state(session_id)
    token_counts = state["token_counts"]
    windowed_history = chat_history_messages[-6:]
    # --- The RAG part is the same as before ---
    query_embedding = embedding_model.embed_query(question)
    vector_search_pipeline = [
        {"$vectorSearch": {"index": "default", "path": "embedding", "queryVector": query_embedding, "numCandidates": 100, "limit": 3}},
        {"$project": {"_id": 0, "text": 1, "source": 1}}
    ]
    results = list(collection.aggregate(vector_search_pipeline))
    context = "\n\n---\n\n".join([doc['text'] for doc in results])
    sources = list(set([doc['source'] for doc in results]))

    # 2. Load chat history
    #chat_history = memory.load_memory_variables({})['history']

    # 3. Create a prompt that includes history
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant. Answer the user's question based on the provided context and the ongoing chat history."),
        ("system", f"Context from documents:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 4. Create a simple chain
    chain = prompt_template | llm

    # 5. Invoke the chain
    response = chain.invoke({"input": question, "history": windowed_history})
    answer = response.content

    # 6. Save the new turn to memory
    #memory.save_context({"input": question}, {"output": answer})

    return answer, sources
# --- App Initialization ---
app = Flask(__name__)
# --- API Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        session_id = data.get('session_id')
        question = data.get('question')

        if not session_id or not question:
            return jsonify({"error": "Missing 'session_id' or 'question'"}), 400

        result = ask_conversational_rag(question, session_id)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "Internal Server Error"}), 500
if __name__ == '__main__':
    app.run(debug=True, port=5000)
