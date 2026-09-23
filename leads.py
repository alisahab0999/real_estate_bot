# leads.py
# WHAT: Incremental lead capture — now extended with real-estate-specific
# fields (budget, timeline, financing status, buying vs renting), merged
# the same non-destructive way as name/email/phone/interest.
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def save_lead(
    client_id: str,
    session_id: str,
    name: str | None = None,
    email: str | None = None,
    phone: str | None = None,
    interest: str | None = None,
    budget_min: float | None = None,
    budget_max: float | None = None,
    preferred_locations: str | None = None,
    timeline: str | None = None,
    financing_status: str | None = None,
    buying_or_renting: str | None = None,
) -> dict:
    existing = (
        supabase.table("leads")
        .select("*")
        .eq("client_id", client_id)
        .eq("session_id", session_id)
        .execute()
    )

    current = existing.data[0] if existing.data else {}

    merged = {
        "client_id": client_id,
        "session_id": session_id,
        "name": name or current.get("name"),
        "email": email or current.get("email"),
        "phone": phone or current.get("phone"),
        "interest": interest or current.get("interest"),
        "budget_min": budget_min if budget_min is not None else current.get("budget_min"),
        "budget_max": budget_max if budget_max is not None else current.get("budget_max"),
        "preferred_locations": preferred_locations or current.get("preferred_locations"),
        "timeline": timeline or current.get("timeline"),
        "financing_status": financing_status or current.get("financing_status"),
        "buying_or_renting": buying_or_renting or current.get("buying_or_renting"),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    result = (
        supabase.table("leads")
        .upsert(merged, on_conflict="client_id,session_id")
        .execute()
    )

    return result.data[0]


def get_lead(client_id: str, session_id: str) -> dict | None:
    response = (
        supabase.table("leads")
        .select("*")
        .eq("client_id", client_id)
        .eq("session_id", session_id)
        .execute()
    )
    return response.data[0] if response.data else None