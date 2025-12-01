from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 1. Load Embeddings
print("Loading Embeddings...")
embeddings = download_hugging_face_embeddings()

# 2. Connect to Pinecone
print("Connecting to Pinecone...")
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 3. Create Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 4. Initialize LLM (Brain)
# Using Gemini 1.5 Pro for better medical reasoning
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=500
)

# 5. Define System Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 6. Build the Chain (Using LCEL - No "langchain.chains" needed)
# This function formats the retrieved docs into a single string


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# The Chain: Retriever -> Format -> Prompt -> LLM -> String Output
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- FLASK ROUTES ---


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(f"User Query: {input}")

    try:
        # Invoke the chain
        response = rag_chain.invoke(input)
        print("Response : ", response)
        return str(response)
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, I encountered an error processing your request."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
