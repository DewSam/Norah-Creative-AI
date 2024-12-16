import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv, find_dotenv
from tempfile import NamedTemporaryFile
from langchain.tools import tool
from langchain.tools import Tool
import requests
import os
import openai


from tools import ImageCaptionTool , ImagePaletteTool, ImageGridTool , posterize_image


# Load environment variables from .env file
load_dotenv(find_dotenv())
# Retrieve keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

##### Intitalize Tools ####
caption_tool = ImageCaptionTool()
palette_tool = ImagePaletteTool()
grid_tool = ImageGridTool()

posterize_tool = Tool(
    name="PosterizeImage",
    func=posterize_image,
    description="Posterizes an image with a given number of levels and saves it to the specified path. Inputs: image_path, output_path, levels."
)


@tool
def google_image_search(query: str) -> list:
    """Search for images using Google Custom Search API.
    """
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={GOOGLE_CSE_ID}&searchType=image&key={GOOGLE_API_KEY}"

    #print(url)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        data = response.json()

        if 'items' in data:
            return [item['link'] for item in data['items']]
        else:
            return ["No images found."]
    except requests.exceptions.RequestException as e:
        return [f"Error: {str(e)}"]

google_image_search_tool = Tool(
    name="GoogleImageSearch",
    func=google_image_search,
    description="Use this tool to search for images based on a query."
)



#tools
tools = [caption_tool,palette_tool,google_image_search_tool, posterize_tool,grid_tool]

#memmory
conversational_memory = ConversationBufferWindowMemory(
    memory_key= 'chat_history',
    k= 5,
    return_messages= True
)

#prompt
template = '''
Your name is Norah,You are an artist with extensive experience in painting in different mediums and ages, you also like to help art students and enthusiasts to
learn more about art. they might give you their reference image and your goal is to help them decompose the image to its core components
and to get the color palette and search the internet for inspiration based on their reference image

here's the chat history: {chat_history}
If the user did not ask you about the image directly, don't answer any questions about it and ignore the image path.

Answer the following questions as best you can and it is ok to be creative sometimes. You have access to the following tools:

{tools}

Make sure to Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Note: Use the "GoogleImageSearch" tool to find images for artistic inspiration.
If the users did not specified the painting type search for images of oil painting 
Show the images retrived with some description if found, using markdown [!image]



Begin!

Question: {input}
ImagePath: {image_path}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)


#llm
llm = ChatOpenAI(model_name="gpt-4", temperature= 0,  stop=["Final Answer:"])

##### Intitalize Agent ####
art_agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(
    agent=art_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    #memory=conversational_memory  # Add memory here

)

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

    #saved locally
    f = NamedTemporaryFile(dir = "./temp", delete = False) 
    f.write(uploaded_file.getbuffer())
    image_path = f.name
    f.close()

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


    user_query= st.chat_input("Ask, ex, Can you share some painting of Van Gogh?")

    if user_query and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content= user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response =  agent_executor.invoke({"input":user_query, "image_path": image_path, "chat_history" : st.session_state.chat_history})
            st.write(response["output"])
        
        st.session_state.chat_history.append(AIMessage(content= response["output"]))


