# agent.py
# WHAT: LangGraph agent with 6 real estate tools — property search,
# need-based matching, showing requests, info-document RAG, lead capture,
# and human escalation — fully MULTI-TENANT.
# WHY: client_id lives in AgentState and is injected into every tool call
# from tools_node, never from the model. Model is openai/gpt-oss-120b.
# Robustness: tools are omitted after 4 tool rounds in a turn, a
# BadRequestError triggers one graceful retry without tools, and
# "degenerate" garbage output (repeated punctuation loops) is detected
# and retried, since Groq returns it as a normal successful response.

import os
import json
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from groq import Groq, BadRequestError
from langgraph.graph import StateGraph, END

from ingest_docs import chroma_client  # single shared Chroma client
from leads import save_lead
from escalations import flag_for_human
from properties import search_properties, find_property_match
from showings import schedule_showing

load_dotenv()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def add_dicts(left: list[dict], right: list[dict]) -> list[dict]:
    return left + right


def retrieve_policy_info(client_id: str, query: str) -> list[dict] | dict:
    # WHY try/except: a missing collection (e.g. docs not ingested yet) must
    # become an honest tool error the model can relay, not a 500 for the user.
    try:
        collection = chroma_client.get_collection(f"policies_{client_id}")
        results = collection.query(query_texts=[query], n_results=2)
    except Exception as e:
        print(f"[retrieve_policy_info] {client_id}: {e}")
        return {
            "error": "no_docs_available",
            "message": "No information documents are available right now.",
        }
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
            "name": "retrieve_policy_tool",
            "description": (
                "Search the agency's information documents (buying and "
                "renting process, fees, application requirements, pet "
                "policy, showing process, office policies). Use this "
                "whenever the customer asks about a process, fee, required "
                "document, or policy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The customer's question, used to search the documents semantically"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_properties_tool",
            "description": "Search available property listings by type, city, price range, or bedrooms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "listing_type": {"type": ["string", "null"], "description": "'sale' or 'rent'"},
                    "city": {"type": ["string", "null"]},
                    "min_price": {"type": ["number", "null"]},
                    "max_price": {"type": ["number", "null"]},
                    "bedrooms": {"type": ["number", "null"]},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_property_match_tool",
            "description": "Search for properties matching a customer's described needs (e.g. '3 bedroom under 400k near downtown with a yard'). If vague, ask ONE clarifying question first (budget, location, bedrooms) before calling this.",
            "parameters": {
                "type": "object",
                "properties": {"needs": {"type": "string"}},
                "required": ["needs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_showing_tool",
            "description": "Book a property showing ONLY after collecting property address, full name, email, phone, preferred date, and preferred time. Never call with missing fields. After calling, tell the customer honestly the request is logged and the agent will confirm — NEVER claim the showing is confirmed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "property_address": {"type": "string"},
                    "customer_name": {"type": "string"},
                    "customer_email": {"type": "string"},
                    "customer_phone": {"type": "string"},
                    "preferred_date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "preferred_time": {"type": "string", "description": "e.g. '2:00 PM'"},
                },
                "required": ["property_address", "customer_name", "customer_email", "customer_phone", "preferred_date", "preferred_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_lead_tool",
            "description": (
                "Save the customer's contact info and real estate needs "
                "incrementally as shared — name, email, phone, what "
                "they're interested in, budget range, timeline, financing "
                "status, and whether buying or renting. Call this as soon "
                "as ANY piece is shared, not all at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                    "interest": {"type": ["string", "null"]},
                    "budget_min": {"type": ["number", "null"]},
                    "budget_max": {"type": ["number", "null"]},
                    "preferred_locations": {"type": ["string", "null"]},
                    "timeline": {"type": ["string", "null"], "description": "e.g. 'ASAP', '3 months', '6+ months'"},
                    "financing_status": {"type": ["string", "null"], "description": "e.g. 'cash', 'pre-approved', 'need financing'"},
                    "buying_or_renting": {"type": ["string", "null"]},
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
                "Flag this conversation for a human agent to review. "
                "Use this when: the customer is frustrated or upset (especially "
                "if they've expressed frustration more than once), the customer "
                "explicitly asks to speak to a real person, the customer asks a "
                "Fair Housing-sensitive question (neighborhood demographics, "
                "safety, whether an area suits certain kinds of people), the "
                "request falls outside what any available tool can resolve "
                "(e.g. price negotiation, contract or legal questions), or "
                "you've been unable to help after a couple of attempts on the "
                "same issue. Always tell the customer honestly that you're "
                "escalating this to a team member — never pretend to resolve "
                "something you can't actually fix."
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
    client_id: str


def is_degenerate_response(text: str) -> bool:
    """Detects the repetition/garbage-output failure mode — long runs of
    '...' or repeated punctuation instead of real words. It's a 'successful'
    API response, not an exception, so we catch it ourselves."""
    if not text:
        return False
    if len(text) > 2000:
        return True
    suspicious_runs = re.findall(r'(?:[.\s…]{2,}){5,}', text)
    if suspicious_runs:
        return True
    return False


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    last_user_index = max(
        (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
    )
    tool_calls_this_turn = sum(
        1 for m in messages[last_user_index:] if m["role"] == "tool"
    )
    allow_tools = tool_calls_this_turn < 4

    base_kwargs = {
        "model": "openai/gpt-oss-120b",
        "messages": state["messages"],
        "temperature": 0.3,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
        "max_tokens": 500,
    }
    create_kwargs = dict(base_kwargs)
    if allow_tools:
        create_kwargs["tools"] = TOOLS_SCHEMA
        create_kwargs["tool_choice"] = "auto"

    message = None
    try:
        response = groq_client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message
        if message.content and not message.tool_calls and is_degenerate_response(message.content):
            raise ValueError("Degenerate output detected")
    except (BadRequestError, ValueError) as e:
        print(f"[agent_node] First attempt failed/degenerate: {e}")
        try:
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["temperature"] = 0.2
            response = groq_client.chat.completions.create(**retry_kwargs)
            message = response.choices[0].message
            if message.content and not message.tool_calls and is_degenerate_response(message.content):
                raise ValueError("Degenerate output on retry too")
        except (BadRequestError, ValueError) as e2:
            print(f"[agent_node] Retry also failed/degenerate: {e2}")
            message = type("obj", (), {
                "content": "Sorry, I ran into a glitch processing that — could you repeat your last message?",
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
    last_message = state["messages"][-1]
    tool_messages = []
    client_id = state["client_id"]
    session_id = state["session_id"]

    for tool_call in last_message["tool_calls"]:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        if name == "retrieve_policy_tool":
            result = retrieve_policy_info(client_id=client_id, **args)
        elif name == "save_lead_tool":
            result = save_lead(client_id=client_id, session_id=session_id, **args)
        elif name == "flag_for_human_tool":
            result = flag_for_human(client_id=client_id, session_id=session_id, **args)
        elif name == "search_properties_tool":
            result = search_properties(client_id=client_id, **args)
        elif name == "find_property_match_tool":
            result = find_property_match(client_id=client_id, **args)
        elif name == "schedule_showing_tool":
            result = schedule_showing(client_id=client_id, session_id=session_id, **args)
        else:
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







# # agent.py
# # WHAT: LangGraph agent with 7 tools — product search, concern-based
# # solution finding, order confirmation, policy RAG, order lookup, lead
# # capture, and human escalation — fully MULTI-TENANT.
# # WHY: client_id lives in AgentState (same pattern as session_id from
# # Phase 5) and is injected into every tool call from tools_node, never
# # from the model. Model is openai/gpt-oss-120b since llama-3.3-70b-
# # versatile is no longer available on this account. Tool calling made
# # more robust against model quirks: tools are OMITTED from the request
# # entirely (not just soft-disabled via tool_choice="none") once one round
# # has happened, a BadRequestError from a hallucinated/malformed tool name
# # triggers one graceful retry without tools, and a NEW check catches
# # "degenerate" garbage output (repeated dots/punctuation loops) that
# # Groq returns as a normal successful response — not an exception — so
# # we have to detect it ourselves after the fact.

# import os
# import json
# import re
# from typing import Annotated, TypedDict
# from dotenv import load_dotenv
# from groq import Groq, BadRequestError
# import chromadb
# from langgraph.graph import StateGraph, END

# from orders import lookup_order
# from leads import save_lead
# from escalations import flag_for_human
# from properties import search_properties, find_property_match
# from showings import schedule_showing

# load_dotenv()

# groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# chroma_client = chromadb.PersistentClient(path="./chroma_store")


# def add_dicts(left: list[dict], right: list[dict]) -> list[dict]:
#     return left + right


# def retrieve_policy_info(client_id: str, query: str) -> list[dict]:
#     collection = chroma_client.get_collection(f"policies_{client_id}")
#     results = collection.query(query_texts=[query], n_results=2)
#     chunks = []
#     for doc, meta, distance in zip(
#         results["documents"][0], results["metadatas"][0], results["distances"][0]
#     ):
#         chunks.append({
#             "text": doc,
#             "source": meta["source"],
#             "distance": round(distance, 3),
#         })
#     return chunks


# TOOLS_SCHEMA = [
#     # {
#     #     "type": "function",
#     #     "function": {
#     #         "name": "search_products_tool",
#     #         "description": (
#     #             "Search the store's product catalog by category and/or maximum "
#     #             "price. Use this whenever the customer asks to find, browse, or "
#     #             "get recommendations for products."
#     #         ),
#     #         "parameters": {
#     #             "type": "object",
#     #             "properties": {
#     #                 "category": {"type": ["string", "null"], "description": "Product category, e.g. 'shoes', 'shirts', 'jackets'. Use null if not filtering by category."},
#     #                 "max_price": {"type": ["number", "null"], "description": "Maximum price filter in USD. Use null if not filtering by price."},
#     #             },
#     #             "required": [],
#     #         },
#     #     },
#     # },
#     # {
#     #     "type": "function",
#     #     "function": {
#     #         "name": "find_solution_tool",
#     #         "description": (
#     #             "Search the product catalog for a product that addresses a "
#     #             "specific customer concern (e.g. 'acne', 'dry skin', 'dark "
#     #             "spots.etc'). Use this when a customer describes a problem they "
#     #             "want solved. If the concern is vague, ask ONE brief "
#     #             "clarifying question first (like skin type or main symptom) you have to be completely sure about the problem "
#     #             "before calling this tool."
#     #         ),
#     #         "parameters": {
#     #             "type": "object",
#     #             "properties": {
#     #                 "concern": {"type": "string", "description": "The customer's stated concern, in a few words"},
#     #             },
#     #             "required": ["concern"],
#     #         },
#     #     },
#     # },
#     # {
#     #     "type": "function",
#     #     "function": {
#     #         "name": "confirm_order_tool",
#     #         "description": (
#     #             "Finalize and log a confirmed order ONLY after the customer "
#     #             "has explicitly agreed to buy AND you have collected ALL "
#     #             "required details: product name, quantity, full name, correct format email, "
#     #             "phone, shipping address, and country. NEVER call this with "
#     #             "missing required fields. After calling this, tell the "
#     #             "customer honestly their order is logged and the team will "
#     #             "follow up with a secure payment link — NEVER claim payment "
#     #             "was processed."
#     #         ),
#     #         "parameters": {
#     #             "type": "object",
#     #             "properties": {
#     #                 "product_name": {"type": "string"},
#     #                 "quantity": {"type": "number"},
#     #                 "customer_name": {"type": "string"},
#     #                 "customer_email": {"type": "string"},
#     #                 "customer_phone": {"type": "string"},
#     #                 "shipping_address": {"type": "string", "description": "Street, city, state, ZIP — NOT including country"},
#     #                 "country": {"type": "string", "description": "The country the order ships to, e.g. 'US', 'Pakistan'"},
#     #             },
#     #             "required": ["product_name", "quantity", "customer_name", "customer_email", "customer_phone", "shipping_address", "country"],
#     #         },
#     #     },
#     # },
    
#     {
#         "type": "function",
#         "function": {
#             "name": "retrieve_policy_tool",
#             "description": (
#                 "Search the store's policy documents (returns, shipping, "
#                 "warranty) for relevant information. Use this whenever the "
#                 "customer asks about returns, refunds, shipping times, "
#                 "shipping costs, warranties, or defective items."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string", "description": "The customer's question, used to search policy documents semantically"},
#                 },
#                 "required": ["query"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "search_properties_tool",
#             "description": "Search available property listings by type, city, price range, or bedrooms.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "listing_type": {"type": ["string", "null"], "description": "'sale' or 'rent'"},
#                     "city": {"type": ["string", "null"]},
#                     "min_price": {"type": ["number", "null"]},
#                     "max_price": {"type": ["number", "null"]},
#                     "bedrooms": {"type": ["number", "null"]},
#                 },
#                 "required": [],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "find_property_match_tool",
#             "description": "Search for properties matching a customer's described needs (e.g. '3 bedroom under 400k near downtown with a yard'). If vague, ask ONE clarifying question first (budget, location, bedrooms) before calling this.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {"needs": {"type": "string"}},
#                 "required": ["needs"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "schedule_showing_tool",
#             "description": "Book a property showing ONLY after collecting property address, full name, email, phone, preferred date, and preferred time. Never call with missing fields. After calling, tell the customer honestly the request is logged and the agent will confirm — NEVER claim the showing is confirmed.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "property_address": {"type": "string"},
#                     "customer_name": {"type": "string"},
#                     "customer_email": {"type": "string"},
#                     "customer_phone": {"type": "string"},
#                     "preferred_date": {"type": "string"},
#                     "preferred_time": {"type": "string"},
#                 },
#                 "required": ["property_address", "customer_name", "customer_email", "customer_phone", "preferred_date", "preferred_time"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "lookup_order_tool",
#             "description": (
#                 "Look up the status and details of a customer's order. "
#                 "REQUIRES both the order number AND the email address used "
#                 "for that order — both must be provided by the customer and "
#                 "must match exactly, or no order will be found. Use this "
#                 "whenever a customer asks about order status, tracking, or "
#                 "delivery. If the customer hasn't provided both order number "
#                 "and email yet, ASK them for whichever is missing before "
#                 "calling this tool."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "order_number": {"type": "string", "description": "The order number, e.g. ORD-1001"},
#                     "email": {"type": "string", "description": "The email address used to place the order"},
#                 },
#                 "required": ["order_number", "email"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "save_lead_tool",
#             "description": (
#                 "Save the customer's contact info and real estate needs "
#                 "incrementally as shared — name, email, phone, what "
#                 "they're interested in, budget range, timeline, financing "
#                 "status, and whether buying or renting. Call this as soon "
#                 "as ANY piece is shared, not all at once."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "name": {"type": ["string", "null"]},
#                     "email": {"type": ["string", "null"]},
#                     "phone": {"type": ["string", "null"]},
#                     "interest": {"type": ["string", "null"]},
#                     "budget_min": {"type": ["number", "null"]},
#                     "budget_max": {"type": ["number", "null"]},
#                     "preferred_locations": {"type": ["string", "null"]},
#                     "timeline": {"type": ["string", "null"], "description": "e.g. 'ASAP', '3 months', '6+ months'"},
#                     "financing_status": {"type": ["string", "null"], "description": "e.g. 'cash', 'pre-approved', 'need financing'"},
#                     "buying_or_renting": {"type": ["string", "null"]},
#                 },
#                 "required": [],
#             },
#         },
#     },
#     # {
#     #     "type": "function",
#     #     "function": {
#     #         "name": "save_lead_tool",
#     #         "description": (
#     #             "Save the customer's contact information (name, email, phone) "
#     #             "and what they're interested in, as they share it during the "
#     #             "conversation. Call this incrementally — as soon as the "
#     #             "customer shares ANY piece of info, save it immediately rather "
#     #             "than waiting to collect everything at once. Only ask for this "
#     #             "info when the customer shows genuine interest in a product or "
#     #             "buying — NEVER ask for contact info during a simple FAQ or "
#     #             "policy question."
#     #         ),
#     #         "parameters": {
#     #             "type": "object",
#     #             "properties": {
#     #                 "name": {"type": ["string", "null"], "description": "Customer's name, if shared"},
#     #                 "email": {"type": ["string", "null"], "description": "Customer's email, if shared"},
#     #                 "phone": {"type": ["string", "null"], "description": "Customer's phone number, if shared"},
#     #                 "interest": {"type": ["string", "null"], "description": "What product/category the customer is interested in"},
#     #             },
#     #             "required": [],
#     #         },
#     #     },
#     # },
#     {
#         "type": "function",
#         "function": {
#             "name": "flag_for_human_tool",
#             "description": (
#                 "Flag this conversation for a human support agent to review. "
#                 "Use this when: the customer is frustrated or upset (especially "
#                 "if they've expressed frustration more than once), the customer "
#                 "explicitly asks to speak to a real person, the request falls "
#                 "outside what any available tool can resolve (e.g. a policy "
#                 "exception request), or you've been unable to help after a "
#                 "couple of attempts on the same issue. Always tell the customer "
#                 "honestly that you're escalating this to a team member — never "
#                 "pretend to resolve something you can't actually fix."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "reason": {"type": "string", "description": "Short reason for the escalation"},
#                     "conversation_summary": {"type": "string", "description": "Brief summary of the issue for the human agent"},
#                 },
#                 "required": ["reason", "conversation_summary"],
#             },
#         },
#     },
# ]


# class AgentState(TypedDict):
#     messages: Annotated[list[dict], add_dicts]
#     session_id: str
#     client_id: str


# def is_degenerate_response(text: str) -> bool:
#     """Detects the repetition/garbage-output failure mode — long runs of
#     '...' or repeated punctuation instead of real words. WHY: this is a
#     'successful' API response, not an exception Groq raises, so we have
#     to catch it ourselves after the fact — a tool never sees this, since
#     the corruption happens in the model's own text generation."""
#     if not text:
#         return False
#     if len(text) > 2000:
#         return True
#     suspicious_runs = re.findall(r'(?:[.\s…]{2,}){5,}', text)
#     if suspicious_runs:
#         return True
#     return False


# def agent_node(state: AgentState) -> dict:
#     messages = state["messages"]
#     last_user_index = max(
#         (i for i, m in enumerate(messages) if m["role"] == "user"), default=-1
#     )
#     tool_calls_this_turn = sum(
#         1 for m in messages[last_user_index:] if m["role"] == "tool"
#     )
#     allow_tools = tool_calls_this_turn < 4

#     base_kwargs = {
#         "model": "openai/gpt-oss-120b",
#         "messages": state["messages"],
#         "temperature": 0.3,
#         # WHY: discourages the model from repeating the same tokens —
#         # directly targets the root cause of the repetition-loop glitch.
#         "frequency_penalty": 0.4,
#         "presence_penalty": 0.3,
#         # WHY: hard ceiling on response length — even if degeneration still
#         # happens, it can never produce a multi-thousand-character wall of
#         # garbage again.
#         "max_tokens": 500,
#     }
#     create_kwargs = dict(base_kwargs)
#     if allow_tools:
#         create_kwargs["tools"] = TOOLS_SCHEMA
#         create_kwargs["tool_choice"] = "auto"

#     message = None
#     try:
#         response = groq_client.chat.completions.create(**create_kwargs)
#         message = response.choices[0].message
#         if message.content and not message.tool_calls and is_degenerate_response(message.content):
#             raise ValueError("Degenerate output detected")
#     except (BadRequestError, ValueError) as e:
#         print(f"[agent_node] First attempt failed/degenerate: {e}")
#         try:
#             retry_kwargs = dict(base_kwargs)
#             retry_kwargs["temperature"] = 0.2
#             response = groq_client.chat.completions.create(**retry_kwargs)
#             message = response.choices[0].message
#             if message.content and not message.tool_calls and is_degenerate_response(message.content):
#                 raise ValueError("Degenerate output on retry too")
#         except (BadRequestError, ValueError) as e2:
#             print(f"[agent_node] Retry also failed/degenerate: {e2}")
#             message = type("obj", (), {
#                 "content": "Sorry, I ran into a glitch processing that — could you repeat your last message?",
#                 "tool_calls": None,
#             })()

#     new_message = {"role": "assistant", "content": message.content}
#     if message.tool_calls:
#         new_message["tool_calls"] = [
#             {
#                 "id": tc.id,
#                 "type": "function",
#                 "function": {"name": tc.function.name, "arguments": tc.function.arguments},
#             }
#             for tc in message.tool_calls
#         ]

#     return {"messages": [new_message]}


# def tools_node(state: AgentState) -> dict:
#     last_message = state["messages"][-1]
#     tool_messages = []
#     client_id = state["client_id"]
#     session_id = state["session_id"]

#     for tool_call in last_message["tool_calls"]:
#         name = tool_call["function"]["name"]
#         args = json.loads(tool_call["function"]["arguments"])

#         # if name == "search_products_tool":
#         #     result = search_products(client_id=client_id, **args)
#         if name == "retrieve_policy_tool":
#             result = retrieve_policy_info(client_id=client_id, **args)
#         elif name == "lookup_order_tool":
#             result = lookup_order(client_id=client_id, **args)
#         elif name == "save_lead_tool":
#             result = save_lead(client_id=client_id, session_id=session_id, **args)
#         elif name == "flag_for_human_tool":
#             result = flag_for_human(client_id=client_id, session_id=session_id, **args)
#         elif name == "search_properties_tool":
#             result = search_properties(client_id=client_id, **args)
#         elif name == "find_property_match_tool":
#             result = find_property_match(client_id=client_id, **args)
#         elif name == "schedule_showing_tool":
#             result = schedule_showing(client_id=client_id, session_id=session_id, **args)
#         else:
#             result = {"error": f"Unknown tool: {name}"}

#         tool_messages.append({
#             "role": "tool",
#             "tool_call_id": tool_call["id"],
#             "content": json.dumps(result, default=str),
#         })

#     return {"messages": tool_messages}


# def should_continue(state: AgentState) -> str:
#     last_message = state["messages"][-1]
#     if last_message.get("tool_calls"):
#         return "tools"
#     return END


# graph_builder = StateGraph(AgentState)
# graph_builder.add_node("agent", agent_node)
# graph_builder.add_node("tools", tools_node)

# graph_builder.set_entry_point("agent")
# graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
# graph_builder.add_edge("tools", "agent")

# agent_graph = graph_builder.compile()