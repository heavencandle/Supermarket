from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from streamlit_pills import pills

st.set_page_config(
    page_title="DocumentDetective",
    page_icon="🐻",
)
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler()
    ]

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

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    return retriever

def save_message(message, role):
    st.session_state["message"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    if role=="assistant":
        with st.chat_message(role, avatar="./image/detective.png"):
            st.markdown(message)
    else:
        with st.chat_message(role, avatar="./image/hamter.png"):
            st.markdown(message)
    if save:
        save_message(message, role)

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

st.session_state["message"]=[]
p_selected = None
file = None

# st.markdown("<style> .stChatMessage  { border: 1px solid #F0F2F6; }</style>", unsafe_allow_html=True)
with st.chat_message("assistant", avatar="./image/detective.png"):
    st.write("Hi, do you need my help?")
    p_selected = pills(label='', label_visibility='collapsed', options=['What can you do?', 'How to use'], index=None, clearable=False)

if p_selected:
    with st.chat_message("assistant", avatar="./image/detective.png"):
        if p_selected == 'What can you do?':
            st.write(f"""
        - **Answer Questions:** Ask Mangle anything about your uploaded documents, and get clear, concise answers.
        - **Summarize Content:** Get quick summaries of sections or entire documents.
        - **Find Key Information:** Let Mangle help you locate important details hidden in your files.
                            """)
        elif p_selected == 'How to use':
            st.write(f"""
        1. **Upload Your Document:** Use the uploader below to add a .txt, .pdf, or .docx file.
        2. **Ask Mangle:** Type your question in the chat, and Mangle will fetch the answer for you.
        3. **Enjoy the Insights:** Whether it's summarizing a section or finding specific information, Mangle is here to help!
            """)
            file=st.file_uploader('', type=["pdf", "txt", "docx"])
        else:
            pass

if file:
    retriever = embed_file(file)
    send_message(f"I've got my magnifying glass ready! Ask away and let's uncover the answers together!", "assistant", save=False)
else:
    st.session_state["message"]=[]

paint_history()
message = st.chat_input("Ask anything about your file...")

if message:
    send_message(message, "human")

    try:
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } 
            | prompt 
            | llm
        )
    except NameError:
        chain = (
            {   
                "context": {},
                "question": RunnablePassthrough()
            } 
            | prompt 
            | llm
        )
    with st.chat_message("assistant", avatar="./image/detective.png"):
        response = chain.invoke(message)

# *Image of Mangle by [Yurang](https://www.instagram.com/yurang_official/?hl=kom).*
        

