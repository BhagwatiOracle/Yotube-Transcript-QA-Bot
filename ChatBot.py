import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
st.title("🎥 YouTube Transcript QA Bot")



video_id = st.text_input("Enter Video ID")


if video_id:

    try:
        transcript_list=YouTubeTranscriptApi.get_transcript(video_id,languages=['en'])

        transcript = ' '.join(chunk['text'] for chunk in transcript_list)
        print(transcript)
    except TranscriptsDisabled:
        st.write("No Caption for this video")

    # Text Splitting

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=splitter.create_documents([transcript])

    # Embedding Generation

    embeddings= HuggingFaceEmbeddings()
    vector_store=FAISS.from_documents(chunks,embeddings)

    # Retrieval
    retriever = vector_store.as_retriever(search_kwargs={'k':4})

        
    # Augmentation
        
    llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')

    prompt=PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )


        # Building Chain

    def format_docs(retrieved_docs):
        context_text='\n\n'.join(doc.page_content for doc in retrieved_docs)
        return context_text


    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser


    query=st.text_input('Ask a question about the video ❓')
    if query:
        response=main_chain.invoke(query)
        st.markdown(f"**🤖 Answer:** {response}")

            


