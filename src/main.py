import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import initialize_agent
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from tempfile import NamedTemporaryFile

from tools import ImageCaptionTool , ImagePaletteTool


# Load environment variables from .env file
_ = load_dotenv(find_dotenv())

##### Intitalize Tools ####
caption_tool = ImageCaptionTool()
palette_tool = ImagePaletteTool()

##### Intitalize Agent ####
tools = [caption_tool,palette_tool]
conversational_memory = ConversationBufferWindowMemory(
    memory_key= 'chat_history',
    k= 5,
    return_messages= True
)

llm = ChatOpenAI(model_name="gpt-4", temperature= 1)
agent = initialize_agent( agent= "chat-conversational-react-description", tools= tools, llm = llm, memory =conversational_memory , verbose = True)


# app config
st.set_page_config(page_title="Norah Chatbot", page_icon="🤖")

#set a title
st.title("Norah - Your Creative AI Assistant")

#set a header
st.header("Hi I am Norah, Please upload your Reference Image to start!")

#File
uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png" ])

if uploaded_file:
    #Display the image to the user
    st.image(uploaded_file,caption="Uploaded Image", use_column_width= True)

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ AIMessage(content="Hello, I am Norah, your AI Art Assistant. How can I help you?") ]

    #conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


    user_query= st.chat_input("please Enter your query")

    if user_query and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content= user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with NamedTemporaryFile(dir = "./temp", delete = False) as f:
            f.write(uploaded_file.getbuffer())
            image_path = f.name
            with st.chat_message("AI"):
                response = agent.run(f"{user_query}, thid is the image path: {image_path}")
                st.write(response)
        
        st.session_state.chat_history.append(AIMessage(content= response))


