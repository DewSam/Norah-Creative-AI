import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from PIL import Image
import base64
import io

from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
_ = load_dotenv(find_dotenv())


# app config
st.set_page_config(page_title="Norah Chatbot", page_icon="🤖")
st.title("Norah - Your Creative AI Assistant")


def get_response(user_query, img_base64, chat_history):

    template = """
    You are a Creative AI Assistant that can provide assistance to art students and ethusists. Use your knowledge in Art and art history to help them inspire and create new ideas. Answer the following questions considering the history of the conversation:

    User uploaded an image encoded in Base64 and provided this text: {user_question}.
    Base64 Image: {img_base64}... (truncated for brevity).
    Chat history: {chat_history}

    Please reference to the image to answer user question
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4")
        
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "img_base64": img_base64[:100]
    })


st.write("Hi I am Norah, Please upload your Reference Image to start!")
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file: 
    image = Image.open(uploaded_file)
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
 
    # Convert image to Base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am Norah, your AI Art Assistant. How can I help you?"),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            #response = get_response(user_query, st.session_state.chat_history)
            response = st.write_stream(get_response(user_query,img_base64, st.session_state.chat_history))

            #response = "I dont know" 
            #st.write(response)

        st.session_state.chat_history.append(AIMessage(content=response))