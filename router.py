from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()

class RouterDiagnosis(BaseModel):
    possible_issues: List[str] = Field(..., alias="possible issues")
    required_solution: List[str] = Field(..., alias="required solution")

parser = PydanticOutputParser(pydantic_object=RouterDiagnosis)
format_instructions = parser.get_format_instructions()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct", 
    task="text-generation",
    
)


model = ChatHuggingFace(llm=llm)


chat_history = []

chat_history.append(SystemMessage(content=f"""
You are a router diagnostic assistant. 
When a user provides router statistics, return a dictionary exactly in this format:

{format_instructions}

The dictionary must include only the two keys:
- "possible issues": a list of network problems
- "required solution": a list of recommended actions

Avoid any extra explanations or commentary.
"""))

def get_router_stats():
    print("📡 Enter router statistics:")
    try:
        signal_strength = int(input("Signal Strength (dBm, e.g. -70): "))
        latency_ms = int(input("Latency (ms): "))
        download_speed_mbps = float(input("Download Speed (Mbps): "))
        upload_speed_mbps = float(input("Upload Speed (Mbps): "))
        packet_loss_percent = float(input("Packet Loss (%): "))
        device_count = int(input("Number of connected devices: "))

        return {
            "signal_strength": signal_strength,
            "latency_ms": latency_ms,
            "download_speed_mbps": download_speed_mbps,
            "upload_speed_mbps": upload_speed_mbps,
            "packet_loss_percent": packet_loss_percent,
            "device_count": device_count
        }
    except ValueError:
        print("❌ Invalid input. Please enter numbers correctly.")
        return get_router_stats()

router_stats = get_router_stats()

chat_history.append(HumanMessage(content=f"Router stats: {router_stats}"))

response = model.invoke(chat_history)
chat_history.append(AIMessage(content=response.content))

try:
    structured = parser.parse(response.content)
    print("\n🧠 Router Diagnosis:")
    pprint(structured.model_dump(by_alias=True), indent=2)
except Exception as e:
    print("❌ Failed to parse structured output.")
    print("Error:", e)
    print("Raw Output:\n", response.content)
