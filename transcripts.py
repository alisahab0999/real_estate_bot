# transcripts.py
# WHAT: Persists every message of every conversation to Supabase.
# WHY: SESSIONS in main.py is in-memory only — it's lost on every restart
# and gives the owner no way to review what the bot actually told a
# customer. This is a separate, append-only log: it never blocks or breaks
# the chat flow if it fails (a logging failure should never become a
# customer-facing error), it just prints and moves on.
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def save_message(
    client_id: str,
    session_id: str,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> None:
    try:
        supabase.table("messages").insert({
            "client_id": client_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        }).execute()
    except Exception as e:
        # WHY: never let transcript logging break an actual customer
        # conversation — this is observability, not the critical path.
        print(f"[transcripts] Failed to save message ({role}, session={session_id}): {e}")


def get_transcript(client_id: str, session_id: str) -> list[dict]:
    """Returns the full ordered transcript for one session — useful for a
    future dashboard, or for manual review via Supabase Studio today."""
    response = (
        supabase.table("messages")
        .select("*")
        .eq("client_id", client_id)
        .eq("session_id", session_id)
        .order("id")
        .execute()
    )
    return response.data