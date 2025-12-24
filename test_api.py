from langchain_nvidia_ai_endpoints import ChatNVIDIA

llm = ChatNVIDIA(model="deepseek-ai/deepseek-r1-0528")

response = llm.invoke("What is 2+2?")
print(response.content)
