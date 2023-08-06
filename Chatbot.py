import utils
import streamlit as st
from streaming import StreamHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

vectorstore_path = "index/realty_seek_vectorstore"

st.set_page_config(page_title="Realty Seek AI", page_icon="💬")

st.header('Jarvis (Realtor Robot)')

# Create OpenAIEmbeddings object using the provided API key
embeddings = OpenAIEmbeddings()

class CustomDataChatbot:

    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self):

        vectordb =  FAISS.load_local(vectorstore_path, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain


    @utils.enable_chat_history
    def main(self):

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            qa_chain = self.setup_qa_chain()

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant",avatar="assets/boom.png"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    
    obj = CustomDataChatbot()
    obj.main()

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)