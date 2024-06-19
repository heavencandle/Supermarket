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
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("DocumentGPT")

st.markdown("""
Welcome!

Use this chatbot to ask questions to Mangle the data detective about your files!
""")

file = st.file_uploader("Upload a .txt .pdfor .docx file", type=["pdf", "txt", "docx"])

if file:
    st.write(file)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore("./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="#",
        chunk_size=4000, # Should be small enough, but not to distort the meaning or context
        chunk_overlap=True,
    )

    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir
    )

    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()


    # map_doc_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system",
    #         """
    #         Use the following portion of a long document to see if any of the text is relevant to answer the question. 
    #         Return any relevant text verbatim.
    #         -------
    #         {context}
    #         """
    #         ),
    #         ("human", "{question}")
    #     ]
    # )

    # map_doc_chain = map_doc_prompt | llm

    # def map_docs(inputs):
    #     documents = inputs['documents']
    #     question = inputs['question']

    #     return "\n\n".join(
    #         map_doc_chain.invoke(
    #             {"context": doc.page_content, "question": question}
    #         ).content 
    #         for doc in documents
    #     )


    # map_chain = { "documents": retriever, "question": RunnablePassthrough() } | RunnableLambda(map_docs)


    # final_prompt = ChatPromptTemplate.from_messages([
    #     (
    #         "system", 
    #         """
    #         Given the following extracted parts of a long document and a question,
    #         create a final answer.
    #         I you don't know the answer, just say that you don't know. 
    #         Dont try to make up an answer.
    #         ------
    #         {context}
    #         """
    #     ), 
    #     ("human", "{question}")
    # ])

    # chain = { "context": map_chain, "question": RunnablePassthrough()} | final_prompt | llm