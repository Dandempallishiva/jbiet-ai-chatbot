import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# -------------------------
# Embedding model
# -------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# Chroma database
# -------------------------

vectorstore = Chroma(
    collection_name="jbiET_documents",
    embedding_function=embedding_model,
    persist_directory="data/chroma_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -------------------------
# Groq LLM
# -------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is not set. Add it to your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)

# -------------------------
# RAG prompt
# -------------------------

rag_prompt = ChatPromptTemplate.from_template("""
You are a helpful chatbot for J.B. Institute of Engineering and Technology (JBIET).

Answer the user's question using ONLY the context provided below.

If the answer cannot be found in the context, say:

"I couldn't find this information in the available JBIET documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
""")

# -------------------------
# Router prompt
# -------------------------

router_prompt = ChatPromptTemplate.from_template("""
Classify the following question into one of two categories:

JBIET
GENERAL

Return ONLY the category name.

Question:
{question}
""")

# -------------------------
# Rewrite prompt
# -------------------------

rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's latest question as a standalone question.

Use the conversation history to understand references such as:
"what about MBA?"
"what is its fee?"
"how about that?"

If the question is already standalone, return it unchanged.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")

# -------------------------
# Conversation history
# -------------------------

chat_history = []


def rewrite_question(question):

    if not chat_history:
        return question

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history
    )

    response = llm.invoke(
        rewrite_prompt.format(
            chat_history=history_text,
            question=question
        )
    )

    return response.content.strip()


def classify_question(question):

    response = llm.invoke(
        router_prompt.format(
            question=question
        )
    )

    return response.content.strip().upper()


# -------------------------
# Main chatbot function
# -------------------------

def chatbot(question):

    standalone_question = rewrite_question(question)

    route = classify_question(standalone_question)

    if "JBIET" in route:

        docs = retriever.invoke(standalone_question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        response = llm.invoke(
            rag_prompt.format(
                context=context,
                question=standalone_question
            )
        )

    else:

        response = llm.invoke(
            standalone_question
        )

    answer = (
        response
        if isinstance(response, str)
        else response.content
    )

    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer