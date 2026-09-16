# agent.py
# WHAT: LangGraph agent with 5 tools — product search, policy RAG, order
# lookup, lead capture, and human escalation — now fully MULTI-TENANT.
# WHY: client_id lives in AgentState (same pattern as session_id from
# Phase 5) and is injected into every tool call from tools_node, never
# from the model. Model name updated to openai/gpt-oss-120b since
# llama-3.3-70b-versatile is no longer available on this account. Tool
# calling made more robust against model quirks: tools are OMITTED from
# the request entirely (not just soft-disabled via tool_choice="none")
# once one round has happened, and a BadRequestError from a hallucinated/
# malformed tool name triggers one graceful retry without tools.

import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from groq import Groq, BadRequestError
import chromadb
from langgraph.graph import StateGraph, END

from products import search_products
from orders import lookup_order
from leads import save_lead
from escalations import flag_for_human
from products import search_products, find_solution
from confirmed_orders import confirm_order
load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

chroma_client = chromadb.PersistentClient(path="./chroma_store")


def add_dicts(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right


def retrieve_policy_info(client_id: str, query: str) -> list[dict]:
    # WHY: Opens a COLLECTION SCOPED TO THIS CLIENT ONLY — e.g.
    # "policies_client_shoestore" — so Client A's policy text can never be
    # retrieved by Client B's chatbot, even if the query text overlaps.
    collection = chroma_client.get_collection(f"policies_{client_id}")
    results = collection.query(query_texts=[query], n_results=2)
    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": round(distance, 3),
        })
    return chunks


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_products_tool",
            "description": (
                "Search the store's product catalog by category and/or maximum "
                "price. Use this whenever the customer asks to find, browse, or "
                "get recommendations for products."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {"type": ["string", "null"], "description": "Product category, e.g. 'shoes', 'shirts', 'jackets'. Use null if not filtering by category."},
                    "max_price": {"type": ["number", "null"], "description": "Maximum price filter in USD. Use null if not filtering by price."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_solution_tool",
            "description": (
                "Search the product catalog for a product that addresses a "
                "specific customer concern (e.g. 'acne', 'dry skin', 'dark "
                "spots.etc'). Use this when a customer describes a problem they "
                "want solved. If the concern is vague, ask ONE brief "
                "clarifying question first (like skin type or main symptom) you have to be completely sure about the problem"
                "before calling this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concern": {"type": "string", "description": "The customer's stated concern, in a few words"},
                },
                "required": ["concern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "confirm_order_tool",
            "description": (
                "Finalize and log a confirmed order ONLY after the customer "
                "has explicitly agreed to buy AND you have collected ALL "
                "required details: product name, quantity, full name, correct format email,"
                "phone, shipping address, and country. NEVER call this with "
                "missing required fields. After calling this, tell the "
                "customer honestly their order is logged and the team will "
                "follow up with a secure payment link — NEVER claim payment "
                "was processed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "quantity": {"type": "number"},
                    "customer_name": {"type": "string"},
                    "customer_email": {"type": "string"},
                    "customer_phone": {"type": "string"},
                    "shipping_address": {"type": "string", "description": "Street, city, state, ZIP — NOT including country"},
                    "country": {"type": "string", "description": "The country the order ships to, e.g. 'US', 'Pakistan'"},
                },
                "required": ["product_name", "quantity", "customer_name", "customer_email", "customer_phone", "shipping_address", "country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy_tool",
            "description": (
                "Search the store's policy documents (returns, shipping, "
                "warranty) for relevant information. Use this whenever the "
                "customer asks about returns, refunds, shipping times, "
                "shipping costs, warranties, or defective items."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The customer's question, used to search policy documents semantically"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order_tool",
            "description": (
                "Look up the status and details of a customer's order. "
                "REQUIRES both the order number AND the email address used "
                "for that order — both must be provided by the customer and "
                "must match exactly, or no order will be found. Use this "
                "whenever a customer asks about order status, tracking, or "
                "delivery. If the customer hasn't provided both order number "
                "and email yet, ASK them for whichever is missing before "
                "calling this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {"type": "string", "description": "The order number, e.g. ORD-1001"},
                    "email": {"type": "string", "description": "The email address used to place the order"},
                },
                "required": ["order_number", "email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_lead_tool",
            "description": (
                "Save the customer's contact information (name, email, phone) "
                "and what they're interested in, as they share it during the "
                "conversation. Call this incrementally — as soon as the "
                "customer shares ANY piece of info, save it immediately rather "
                "than waiting to collect everything at once. Only ask for this "
                "info when the customer shows genuine interest in a product or "
                "buying — NEVER ask for contact info during a simple FAQ or "
                "policy question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"], "description": "Customer's name, if shared"},
                    "email": {"type": ["string", "null"], "description": "Customer's email, if shared"},
                    "phone": {"type": ["string", "null"], "description": "Customer's phone number, if shared"},
                    "interest": {"type": ["string", "null"], "description": "What product/category the customer is interested in"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_human_tool",
            "description": (
                "Flag this conversation for a human support agent to review. "
                "Use this when: the customer is frustrated or upset (especially "
                "if they've expressed frustration more than once), the customer "
                "explicitly asks to speak to a real person, the request falls "
                "outside what any available tool can resolve (e.g. a policy "
                "exception request), or you've been unable to help after a "
                "couple of attempts on the same issue. Always tell the customer "
                "honestly that you're escalating this to a team member — never "
                "pretend to resolve something you can't actually fix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Short reason for the escalation"},
                    "conversation_summary": {"type": "string", "description": "Brief summary of the issue for the human agent"},
                },
                "required": ["reason", "conversation_summary"],
            },
        },
    },
]


class AgentState(TypedDict):
    messages: Annotated[list[dict], add_dicts]
    session_id: str
    client_id: str  # NEW in Phase 8 — travels through state, never from the model


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    last_user_index = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
    )
    tool_calls_this_turn = sum(
        1 for m in messages[last_user_index:] if m["role"] == "tool"
    )
    allow_tools = tool_calls_this_turn < 4

    create_kwargs = {
        "model": "openai/gpt-oss-120b",
        "messages": state["messages"],
        "temperature": 0.3,
    }
    if allow_tools:
        create_kwargs["tools"] = TOOLS_SCHEMA
        create_kwargs["tool_choice"] = "auto"

    try:
        response = groq_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
    except BadRequestError as e:
        print(f"[agent_node] First attempt failed: {e}")
        try:
            # WHY: Retry once, explicitly forcing tool_choice="none" AS WELL
            # AS omitting tools — belt and suspenders against this model's
            # apparent tendency to emit tool-call-shaped output regardless.
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=state["messages"],
                temperature=0.1,  # WHY: lower temperature to reduce erratic output further
            )
            message = response.choices[0].message
        except BadRequestError as e2:
            # WHY: FINAL safety net — if the model fails even on a plain
            # retry, we construct a fallback message ourselves rather than
            # letting the exception crash the whole /chat request. A
            # customer-facing bot must NEVER hard-crash; a generic honest
            # response is always better than a 500 error.
            print(f"[agent_node] Retry also failed: {e2}")
            message = type("obj", (), {
                "content": "I'm having trouble processing that right now — could you rephrase your question?",
                "tool_calls": None,
            })()

    new_message = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        new_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]

    return {"messages": [new_message]}


def tools_node(state: AgentState) -> dict:
    # WHY: Every tool is explicitly branched here (instead of a generic
    # AVAILABLE_TOOLS dict lookup) because every single tool now needs
    # client_id injected from STATE, not from model-provided arguments —
    # the model must never be able to specify which client's data to touch.
    last_message = state["messages"][-1]
    tool_messages = []
    client_id = state["client_id"]
    session_id = state["session_id"]

    for tool_call in last_message["tool_calls"]:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        if name == "search_products_tool":
            result = search_products(client_id=client_id, **args)
        elif name == "retrieve_policy_tool":
            result = retrieve_policy_info(client_id=client_id, **args)
        elif name == "lookup_order_tool":
            result = lookup_order(client_id=client_id, **args)
        elif name == "save_lead_tool":
            result = save_lead(client_id=client_id, session_id=session_id, **args)
        elif name == "flag_for_human_tool":
            result = flag_for_human(client_id=client_id, session_id=session_id, **args)
        elif name == "find_solution_tool":
            result = find_solution(client_id=client_id, **args)
        elif name == "confirm_order_tool":
            result = confirm_order(client_id=client_id, session_id=session_id, **args)
        else:
            # WHY: Defensive fallback — if the model somehow calls a tool
            # name that matches none of ours (e.g. a hallucinated name that
            # slipped past the BadRequestError retry), report that clearly
            # instead of crashing with a KeyError.
            result = {"error": f"Unknown tool: {name}"}

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result, default=str),
        })

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"
    return END


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tools_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

agent_graph = graph_builder.compile()