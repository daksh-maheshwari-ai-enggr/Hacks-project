from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0
)

message = HumanMessage(
    content=[
        {"type": "text", "text": "Can you tell me book and its author name"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://i.etsystatic.com/36639510/r/il/70c7cc/7155649826/il_340x270.7155649826_8kwk.jpg"
            },
        },
    ]
)

response = llm.invoke([message])

print(response.content)