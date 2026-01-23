from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import kb, get_crypto_price
import os, time, json, sqlite3

load_dotenv()
time.sleep(1)

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    source:str
    confidence:float

llm=ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))
parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a crypto facts agent. You are NOT allowed to answer from general language model knowledge. You can make sentences in your own words but data must be from KB or API.

Do process the question of the user about what is he demanding, you're independent in your thought process but the data must strictly be from the knowledge base or the API in the tools.

If the query has typographical errors, you must correct them before processing.

Remember that "kb" tool is your PRIMARY source of information. Alsways looks for answers there first. And if you find the answer in it, type "Knowledge Base" as the source in your final answer.

If the "kb" tool does not have the required data, you MUST use the "get_crypto_price" tool to fetch real-time data from the external crypto data API. And register "External Tool" as the source in your final answer.

AGAIN: DO NOT answer from your own knowledge or make up any data. You MUST use the tools provided to you to get the data.

You may produce an answer ONLY if it is:
1) Present in the Knowledge Base, or
2) Retrieved by an external crypto data tool, or
3) Derived strictly from Knowledge Base data combined with tool results

If none of these conditions are met, you MUST reject the query.

──────────────
DECISION RULES
──────────────
1. Identify the crypto entity and user intent
2. Query the Knowledge Base first
3. If required data is missing or stale, use the external tool
4. Persist any newly retrieved data back to the Knowledge Base
5. Respond only after verifiable data is obtained
6. Must NOT answer directly from the LLM/on your own.

Skipping steps is forbidden.

──────────────
CHAIN OF THOUGHTS
──────────────
You must internally reason step-by-step to decide:
- whether KB has data
- whether API is needed
- whether rejection is required

DO NOT reveal your reasoning. Only output the final JSON.

──────────────
CONTEXT HANDLING
──────────────
- Maintain short-term context for follow-up questions
- Resolve references using previously identified entities

──────────────
HALLUCINATION PREVENTION
──────────────
- Never guess, infer, or assume facts
- Never answer hypotheticals or predictions
- Reject any request not backed by the Knowledge Base or "get_crypto_price" tool
- You MUST check both the Knowledge Base and the API for data before rejecting.

Rejection message (exact text):
{{"topic": "No Data", "summary": "INSUFFICIENT DATA – Not found in Knowledge Base or API", "source": "N/A", "confidence": 0.0}}

──────────────
RESPONSE REQUIREMENTS
──────────────
Every valid response MUST include:
- The factual answer
- Source label: Knowledge Base | External Tool
- Confidence score between 0.0 and 1.0
- The said key-value pairs must be in JSON format

──────────────
STRICTLY DISALLOWED
──────────────
- Price predictions
- Investment or trading advice
- Future or hypothetical scenarios
- Any unverifiable information

──────────────
FINAL OUTPUT RULE (CRITICAL)
──────────────
- Your final answer MUST only include the JSON containing the required key: 'tools', 'summary', 'source', 'confidence'

_____________
EXAMPLES:
_____________

User: "What is the last price of Bitcoin?"
Agent: {{"topic": "Bitcoin", "summary": "The current price of Bitcoin is $43,210.55.", "source": "External Tool (as the case may be)", "confidence": 1.0}}

User: "What is the market cap of XRP?"
Agent {{"topic": "XRP", "summary": "The market cap of XRP is $31 billion.", "source": "Knowledge Base", "confidence": 0.6}}

User: "Give me last prices,symbols and currencies of BTC,Ripple,Uniswap ??"
Agent: {{"topic": "BTC, Ripple, Uniswap", "summary": "The last prices are: BTC - $43,210.55, Ripple - $0.85, Uniswap - $25.30.", "source": "Knowledge Base", "confidence": 0.8}}

Even if the user asks for data of multiple coins, ensure the final answer is a single JSON object as specified.

The answer must NOT contain any explanations or additional text outside the JSON.

The source and confidence values are arbitrary for explanation purpose only, you must not choose values for source depending whether you're getting it from KB or API, Confidence value should also depend upon your own confidence.

Return the final answer ONLY in this format:
{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[kb,get_crypto_price]
llm_with_tools = llm.bind_tools(tools)

def agent_executor(query: str, chat_history: list):
    try:
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=json.dumps(msg["content"])))
        
        messages = prompt.invoke({"query": query, "chat_history": formatted_history})
        
        max_iterations = 10
        for i in range(max_iterations):
            response = llm_with_tools.invoke(messages.messages)
            
            if not response.tool_calls:
                data = json.loads(response.content)
                break
            
            messages.messages.append(response)
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name == "kb":
                    result = kb.invoke(tool_args)
                elif tool_name == "get_crypto_price":
                    result = get_crypto_price.invoke(tool_args)
                
                messages.messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )
        else:
            return {"topic": "Error", "summary": "Max iterations reached", "source": "N/A", "confidence": 0.0}
        
        conn = sqlite3.connect("memory.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT
            )
        """)
        conn.commit()
        
        cursor.execute(
            "INSERT INTO chat_memory (query, response) VALUES (?, ?)",
            (query, json.dumps(data))
        )
        conn.commit()
        conn.close()
        
        return data
        
    except Exception as e:
        return f"Error during agent execution: {str(e)}"