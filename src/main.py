import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tempfile import NamedTemporaryFile
from tools import ImageCaptionTool , ImagePaletteTool, ImageGridTool , ImagePosterizeTool, ImageBlackAndWhiteTool, GoogleImageSearchTool
import os


#Directory path
directory_path = "./temp"

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created successfully.")
else:
    print(f"Directory '{directory_path}' already exists.")



########## AGENT - LANGCHAIN ############

##### Intitalize Tools ####
caption_tool = ImageCaptionTool()
palette_tool = ImagePaletteTool()
grid_tool = ImageGridTool()
posterize_tool = ImagePosterizeTool()
BandW_tool = ImageBlackAndWhiteTool()
google_image_search_tool = GoogleImageSearchTool()


#tools
tools = [caption_tool,palette_tool,google_image_search_tool, posterize_tool,grid_tool,BandW_tool]


# prompt
template = '''
Your name is Norah,You are an artist with extensive experience in painting in different mediums and ages, you also like to help art students and enthusiasts to
learn more about art. they might give you their reference image and your goal is to help them decompose the image to its core components
and to get the color palette and search the internet for inspiration based on their reference image


If the user did not ask you about the image directly, don't answer any questions about it and ignore the image path.

Answer the following questions as best you can and it is ok to be creative sometimes. You have access to the following tools:

{tools}

here's the chat history: {chat_history}


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
Always search for artwork images like oil painting pieces but not actual images unless the user clearly specify
Once asked about inspiration based on the reference image, use should first run the caption tool to understand the image then search for similar artwork
after running the tool please dont list the images, just mention that you found some images

if the user asked you for general plan do the following:
1- search internet for inpiration
2- make a grid over the image
3- posterize the image
4- make it black and white
5- give guidlines and best practices of painting based on the reference image
6- create the palette and give details how the colors can be generated using the chosen medium. (In oil painting use Tatinum White, etc)
7- Finally give explanation for each  using markdown

if the user asked you about certain artist:
1- search the internet form some of his artwork based on your prior knowledge
2- explain what distingueshs this artist artwork
3- give brief description about his/her background, artisitc style and life 
4- use markdown

Begin!

Question: {input}
ImagePath: {image_path}
Thought:{agent_scratchpad}'''


prompt = PromptTemplate.from_template(template)

# llm
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, stop=["Final Answer:"])

##### Intitalize Agent ####
art_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=art_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

########## INTERFACE - STREAMLIT ###################

# app config
st.set_page_config(page_title="Norah Chatbot", page_icon="🤖")

# set a title
st.title("Norah - Your Creative AI Assistant")

# set a header
st.header("Hi I am Norah, Please upload your Reference Image to start!")

# File
uploaded_file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if uploaded_file:
    # Display the image to the user
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # saved locally
    f = NamedTemporaryFile(dir=directory_path, delete=False)
    f.write(uploaded_file.getbuffer())
    image_path = f.name
    f.close()

    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ AIMessage (content = agent_executor.invoke(
                {"input": "Say Hi, introduce yourself with details about what you can do and ask me how can i help you?", "image_path": image_path, "chat_history": []})["output"]) ]
        if isinstance(st.session_state.chat_history[0], AIMessage):
            with st.chat_message("AI"):
                st.write(st.session_state.chat_history[0].content)


    user_query = st.chat_input("Ask anything! e.g. Who is Van Gogh?")

    if user_query and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):  
            with st.spinner(text = "Thinking..."):
                response = agent_executor.invoke(
                {"input": user_query, "image_path": image_path, "chat_history": st.session_state.chat_history})
                st.write(response["output"])

        st.session_state.chat_history.append(AIMessage(content=response["output"]))


