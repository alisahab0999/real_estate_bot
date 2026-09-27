# # clients.py
# # WHAT: Resolves an API key (sent by a client's widget) into a client_id.
# # WHY: This is the SINGLE gatekeeping point for multi-tenancy — every
# # request must pass through this before touching any client-specific data.
# # If the key is invalid, we stop here rather than letting a bad request
# # reach any tool.

# import os
# from dotenv import load_dotenv
# from supabase import create_client, Client

# load_dotenv()

# supabase: Client = create_client(
#     os.environ["SUPABASE_URL"],
#     os.environ["SUPABASE_SERVICE_KEY"],
# )


# def resolve_client(api_key: str) -> dict | None:
#     """Look up a client by their API key. Returns None if invalid —
#     the caller (main.py) must treat that as a hard rejection, not a
#     fallback to some default client."""
#     response = (
#         supabase.table("clients")
#         .select("*")
#         .eq("api_key", api_key)
#         .execute()
#     )
#     return response.data[0] if response.data else None
# clients.py
# WHAT: Resolves an API key (sent by a client's widget) into a client_id.
# WHY: This is the SINGLE gatekeeping point for multi-tenancy — every
# request must pass through this before touching any client-specific data.
# If the key is invalid, we stop here rather than letting a bad request
# reach any tool.

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def resolve_client(api_key: str) -> dict | None:
    """Look up a client by their API key. Returns None if invalid —
    the caller (main.py) must treat that as a hard rejection, not a
    fallback to some default client."""
    response = (
        supabase.table("clients")
        .select("*")
        .eq("api_key", api_key)
        .execute()
    )
    return response.data[0] if response.data else None


def get_groq_api_key(client_id: str) -> str:
    """Returns this client's own Groq API key if one is set, otherwise the
    shared platform default. WHY: isolates each client's LLM usage so one
    client's traffic spike can't burn through another client's rate limit
    (the 'noisy neighbor' problem) — each agency effectively gets its own
    lane."""
    response = (
        supabase.table("clients")
        .select("groq_api_key")
        .eq("client_id", client_id)
        .execute()
    )
    key = response.data[0].get("groq_api_key") if response.data else None
    return key or os.environ["GROQ_API_KEY"]