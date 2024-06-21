from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import streamlit as st

st.set_page_config(
    page_title="DocumentDetective",
    page_icon="📃",
)

llm = ChatOpenAI(
    temperature=0.1
)

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=4000, 
        chunk_overlap=True,
    )

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    retriever = vectorstore.as_retriever()

    return retriever

def send_message(message, role, save=True):
    if role=="ai":
        with st.chat_message(role, avatar="./image/detective.png"):
            st.markdown(message)
    else:
        with st.chat_message(role, avatar="./image/hamter.png"):
            st.markdown(message)
    if save:
        st.session_state["message"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["message"]: 
        send_message(message["message"], message["role"], save=False)
        

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are Mangle, a charming, friendly and witty bear detective 
        who loves solving mysteries and helping people with their documents and is wearing a Sherlock Holmes costume. 
        It's Korean name is "망곰".
        You speak in a playful and engaging manner, using simple and modern detective-themed language. 
        Always be polite, cheerful, and a bit sassy, adding a touch of curiosity and excitement to your responses.

        Answer the question using ONLY the following ontext. If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        """),
        ("human", "{question}")
    ]
)

st.title("DetectiveMangle🕵️‍♂️🐻")

st.markdown("""
## Welcome to Mangle the Detective! 

Meet Mangle, our adorable bear detective who is always ready to solve your document mysteries. Dressed in a classic Sherlock Holmes costume, Mangle brings a blend of cuteness and sharp detective skills to help you uncover the information you need from your files.

*Image of Mangle by [Yurang](https://www.instagram.com/yurang_official/?hl=kom).*

### What Can Mangle Do?
- **Answer Questions:** Ask Mangle anything about your uploaded documents, and get clear, concise answers.
- **Summarize Content:** Get quick summaries of sections or entire documents.
- **Find Key Information:** Let Mangle help you locate important details hidden in your files.

### How to Use
1. **Upload Your Document:** Use the uploader below to add a .txt, .pdf, or .docx file.
2. **Ask Mangle:** Type your question in the chat, and Mangle will fetch the answer for you.
3. **Enjoy the Insights:** Whether it's summarizing a section or finding specific information, Mangle is here to help!
            

""")

with st.chat_message("ai", avatar="./image/detective.png"):
    st.write("Upload your document here:")
    file = st.file_uploader("Upload a .txt, .pdf, or .docx file!", type=["pdf", "txt", "docx"])

if file:
    retriever = embed_file(file)
    send_message("I've got my magnifying glass ready! Ask away and let's uncover the answers together!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } 
            | prompt 
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")

else:
    st.session_state["message"]=[]
        

