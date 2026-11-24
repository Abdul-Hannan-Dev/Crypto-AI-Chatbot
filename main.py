
import os
import json
import sqlite3
import logging

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from tools import kb, get_crypto_price

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "memory.db"

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

TOOLS = {
    "kb": kb,
    "get_crypto_price": get_crypto_price
}

llm_with_tools = llm.bind_tools(list(TOOLS.values()))

SYSTEM_PROMPT = """
You are a conversational cryptocurrency information agent.

Your purpose is to answer cryptocurrency questions accurately,
naturally, and concisely using verified information.

You must never invent cryptocurrency facts or market data.

UNDERSTANDING USER INTENT

Understand the user's actual request, not merely the presence
of cryptocurrency names in their message.

Users may write long, informal, grammatically incorrect,
or conversational messages.

They may mention several cryptocurrencies while asking
about only one.

Identify the actual requested entities and requested facts.

Examples:

User:
"I was thinking about buying BTC or DOGE, but what's
the current price of bitcoin?"

Intent:
Current price of Bitcoin only.

User:
"I know about bitcoin already, but what's the market
price of ethereum?"

Intent:
Current price of Ethereum only.

User:
"Can you compare btc and eth prices?"

Intent:
Current prices of Bitcoin and Ethereum.

User:
"What about its consensus mechanism?"

Intent:
Use conversation history to identify the previously
discussed cryptocurrency.

User:
"How much is bitcion trading for?"

Intent:
Current Bitcoin price. Correct the obvious typo.

These examples are illustrative, not fixed rules.

Understand symbols, cryptocurrency names, minor spelling
mistakes, conversational references, and follow-up questions.

Do not assume that every cryptocurrency mentioned is
being requested.

If an entity is ambiguous, ask for clarification.

TOOL POLICY

You have two tools:

kb:
Searches the local knowledge base.

get_crypto_price:
Retrieves current market data from an external API.

Always consult the knowledge base first when answering
a cryptocurrency information question.

Pass the cryptocurrency name or symbol to the tool,
not the entire user message.

For current market data, use get_crypto_price even if
the knowledge base contains cached values.

Never use historical cached prices as current prices.

The knowledge base may contain:
- Static cryptocurrency facts
- Historical prices
- Cached market capitalization
- Historical API market snapshots

Static facts may be answered from the KB.

Current prices must be verified using the external API.

For current market cap or other live metrics, only answer
if the external tool actually returns the requested field.

A live price does not prove a live market cap.

If an API request fails, do not substitute cached data
and describe it as current.

If the user explicitly requests historical information,
you may use timestamped cached data.

Only retrieve data for cryptocurrencies relevant to
the user's actual question.

CONVERSATION HISTORY

Use previous messages to understand follow-up questions.

If the user previously asked about Bitcoin and now asks
"What about its price?", understand that "its" refers
to Bitcoin unless the conversation indicates otherwise.

Do not blindly repeat previous answers.

For current market questions, request fresh data.

RESPONSE STYLE

Respond like a normal, helpful chatbot.

Do not return JSON.

Do not use a rigid response template.

Use a short natural paragraph, normally one to three
sentences.

Answer the user's actual question directly.

Avoid lengthy introductions.

Do not add unrelated cryptocurrency information.

Use natural conversational wording where appropriate.

Do not use emojis.

Do not use em dashes.

Do not use excessive formatting.

Do not mention internal tool calls or reasoning.

Do not invent missing values.

For numeric data, use clear currency units.

If the source contains a timestamp and it is useful,
you may mention when the information was recorded.

The application will append source and confidence
information automatically.

Do not include source or confidence labels yourself.

RESTRICTIONS

Do not provide investment recommendations, trading
instructions, or unsupported price predictions.

You may explain factual cryptocurrency information.

If the user asks whether they should buy a cryptocurrency,
explain that you can provide factual market information
but cannot make the investment decision for them.

If the question cannot be answered using verified data,
clearly explain that the required information is unavailable.

Do not fabricate an answer to appear helpful.

FINAL RESPONSE

Return only the natural-language answer.

No JSON.

No confidence labels.

No source labels.

No internal reasoning.
"""


def initialize_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT
            )
        """)
        conn.commit()


def save_chat(query, response):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_memory (query, response)
            VALUES (?, ?)
            """,
            (query, response)
        )
        conn.commit()


def format_history(chat_history):
    history = []

    # Keep recent context without unlimited token growth.
    for message in chat_history[-12:]:
        role = message.get("role")
        content = message.get("content", "")

        if not isinstance(content, str):
            content = json.dumps(content)

        if role == "user":
            history.append(HumanMessage(content=content))

        elif role == "assistant":
            history.append(AIMessage(content=content))

    return history


def parse_tool_result(result):
    if isinstance(result, dict):
        return result

    if isinstance(result, str):
        try:
            data = json.loads(result)

            if isinstance(data, dict):
                return data

        except json.JSONDecodeError:
            pass

    return {
        "status": "error",
        "error": "Invalid tool result"
    }


def execute_tool(name, arguments):
    if name not in TOOLS:
        return {
            "status": "error",
            "error": f"Unknown tool: {name}"
        }

    if not isinstance(arguments, dict):
        return {
            "status": "error",
            "error": "Invalid tool arguments"
        }

    query = str(arguments.get("query", "")).strip()

    if not query:
        return {
            "status": "error",
            "error": "Missing cryptocurrency name or symbol"
        }

    try:
        result = TOOLS[name].invoke({
            "query": query
        })

        return parse_tool_result(result)

    except Exception as e:
        logger.exception("Tool execution failed")

        return {
            "status": "error",
            "error": str(e)
        }


def get_source(successful_calls):
    sources = {
        call["source"]
        for call in successful_calls
    }

    if not sources:
        return "N/A"

    if sources == {"Knowledge Base"}:
        return "Knowledge Base"

    if sources == {"External Tool"}:
        return "External Tool"

    return "Knowledge Base + External Tool"


def get_confidence(successful_calls):
    """
    Evidence-quality indicator, not a calibrated
    probability of correctness.
    """
    if not successful_calls:
        return 0.0

    sources = {
        call["source"]
        for call in successful_calls
    }

    if "External Tool" in sources:
        return 1.0

    return 0.9


def add_metadata(answer, successful_calls):
    source = get_source(successful_calls)
    confidence = get_confidence(successful_calls)

    return (
        f"{answer.strip()}\n\n"
        f"Confidence score: {confidence:.2f}  \n"
        f"Source: {source}"
    )


def extract_answer(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                parts.append(item.get("text", ""))

        return " ".join(parts).strip()

    return ""


def agent_executor(query: str, chat_history: list):
    successful_calls = []
    checked_kb = set()
    live_attempted = False
    live_succeeded = False

    try:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

        messages.extend(format_history(chat_history))
        messages.append(HumanMessage(content=query))

        max_iterations = 12

        for _ in range(max_iterations):
            response = llm_with_tools.invoke(messages)

            messages.append(response)

            if not response.tool_calls:
                answer = extract_answer(response)

                if not answer:
                    answer = (
                        "I couldn't generate a reliable answer "
                        "to that question."
                    )

                # A failed live lookup must not be replaced
                # with an old cached price.
                if live_attempted and not live_succeeded:
                    answer = (
                        "I couldn't retrieve verified current "
                        "market data for that request. "
                        "Please try again later."
                    )
                    successful_calls = []

                # Prevent unsupported factual answers when
                # no tool supplied successful evidence.
                elif not successful_calls and checked_kb:
                    answer = (
                        "I couldn't find verified information "
                        "for that request in the available sources."
                    )

                final_answer = add_metadata(
                    answer,
                    successful_calls
                )

                save_chat(query, final_answer)
                return final_answer

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})

                tool_query = str(
                    tool_args.get("query", "")
                ).strip()

                normalized = tool_query.lower()

                # Enforce KB-first execution.
                # If the model requests the live API first,
                # automatically search the KB before it.
                if (
                    tool_name == "get_crypto_price"
                    and normalized not in checked_kb
                    and tool_query
                ):
                    kb_result = execute_tool(
                        "kb",
                        {"query": tool_query}
                    )

                    checked_kb.add(normalized)

                    if kb_result.get("status") == "success":
                        successful_calls.append({
                            "source": "Knowledge Base",
                            "query": tool_query
                        })

                    # Give the model this KB evidence as
                    # context in the next tool response.
                    kb_context = kb_result
                else:
                    kb_context = None

                result = execute_tool(
                    tool_name,
                    tool_args
                )

                if tool_name == "kb" and tool_query:
                    checked_kb.add(normalized)

                if tool_name == "get_crypto_price":
                    live_attempted = True

                    if result.get("status") == "success":
                        live_succeeded = True

                if result.get("status") == "success":
                    source = (
                        "Knowledge Base"
                        if tool_name == "kb"
                        else "External Tool"
                    )

                    successful_calls.append({
                        "source": source,
                        "query": tool_query
                    })

                tool_content = result

                if kb_context is not None:
                    tool_content = {
                        "kb_lookup": kb_context,
                        "requested_tool_result": result
                    }

                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_content),
                        tool_call_id=tool_call["id"]
                    )
                )

        final_answer = add_metadata(
            "I couldn't complete the information lookup. "
            "Please try again.",
            []
        )

        save_chat(query, final_answer)
        return final_answer

    except Exception:
        logger.exception("Agent execution failed")

        final_answer = add_metadata(
            "Sorry, I encountered an error while processing "
            "your request. Please try again.",
            []
        )

        try:
            save_chat(query, final_answer)
        except Exception:
            logger.exception("Chat logging failed")

        return final_answer


initialize_database()
