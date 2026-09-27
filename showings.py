# # showings.py
# # WHAT: Logs a property showing REQUEST (never a confirmation) and notifies
# # the client's n8n webhook (if configured).
# # WHY: Hard validation lives in CODE, not just the prompt — the model can
# # forget rules, code can't.
# import os
# import re
# from datetime import datetime, date
# from dotenv import load_dotenv
# from supabase import create_client, Client

# load_dotenv()
# supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# def schedule_showing(client_id: str, session_id: str, property_address: str,
#                      customer_name: str, customer_email: str, customer_phone: str,
#                      preferred_date: str, preferred_time: str) -> dict:
#     # 1. Email format
#     if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+$", customer_email.strip()):
#         return {"error": "invalid_email",
#                 "message": "That email address doesn't look valid. Please double-check it."}

#     # 2. Date must be a real, future (or today) calendar date: YYYY-MM-DD
#     try:
#         d = datetime.strptime(preferred_date.strip(), "%Y-%m-%d").date()
#         if d < date.today():
#             return {"error": "invalid_date",
#                     "message": "That date is in the past. Please pick a future date."}
#     except ValueError:
#         return {"error": "invalid_date",
#                 "message": "I couldn't read that date. Please give a specific calendar date."}

#     # 3. The property must exist for THIS client and still be available
#     match = (
#         supabase.table("properties")
#         .select("address")
#         .eq("client_id", client_id)
#         .eq("status", "available")
#         .ilike("address", f"%{property_address.strip()}%")
#         .execute()
#     )
#     if not match.data:
#         return {"error": "property_not_found",
#                 "message": "I can't find an available listing at that address."}

#     result = (
#         supabase.table("showing_requests")
#         .insert({
#             "client_id": client_id,
#             "session_id": session_id,
#             "property_address": match.data[0]["address"],
#             "customer_name": customer_name,
#             "customer_email": customer_email.strip(),
#             "customer_phone": customer_phone,
#             "preferred_date": preferred_date,
#             "preferred_time": preferred_time,
#             "status": "pending",
#         })
#         .execute()
#     )
#     row = result.data[0]

#     client_resp = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
#     webhook_url = client_resp.data[0].get("n8n_webhook_url") if client_resp.data else None
#     if webhook_url:
#         try:
#             import httpx
#             httpx.post(webhook_url, json=row, timeout=5)
#         except Exception as e:
#             print(f"[schedule_showing] Failed to notify n8n: {e}")

#     return row









# # # showings.py
# # # WHAT: Logs a property showing request and notifies the client's n8n
# # # webhook (if configured) — replaces confirmed_orders.py for real estate.
# # import os
# # from dotenv import load_dotenv
# # from supabase import create_client, Client

# # load_dotenv()
# # supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# # def schedule_showing(client_id: str, session_id: str, property_address: str,
# #                       customer_name: str, customer_email: str, customer_phone: str,
# #                       preferred_date: str, preferred_time: str) -> dict:
# #     result = (
# #         supabase.table("showing_requests")
# #         .insert({
# #             "client_id": client_id,
# #             "session_id": session_id,
# #             "property_address": property_address,
# #             "customer_name": customer_name,
# #             "customer_email": customer_email,
# #             "customer_phone": customer_phone,
# #             "preferred_date": preferred_date,
# #             "preferred_time": preferred_time,
# #             "status": "pending",
# #         })
# #         .execute()
# #     )
# #     order_row = result.data[0]

# #     client_resp = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
# #     webhook_url = client_resp.data[0].get("n8n_webhook_url") if client_resp.data else None
# #     if webhook_url:
# #         try:
# #             import httpx
# #             httpx.post(webhook_url, json=order_row, timeout=5)
# #         except Exception as e:
# #             print(f"[schedule_showing] Failed to notify n8n: {e}")

# #     return order_row


# showings.py
# WHAT: Logs a property showing REQUEST (never a confirmation) and notifies
# the client's n8n webhook (if configured).
# WHY: Hard validation lives in CODE, not just the prompt — the model can
# forget rules, code can't. This version adds a double-booking check: two
# requests for the same property, same date, within a buffer window of each
# other are treated as a conflict and rejected before they ever reach the
# owner as two separate "surprise" bookings.

import os
import re
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# Two showings on the same property, same day, within this many minutes of
# each other are treated as a conflict. 60 min covers a typical showing slot
# plus buffer; adjust per client later if needed.
SHOWING_BUFFER_MINUTES = 60

# Statuses that count as "still active" when checking for conflicts — a
# cancelled or declined request should never block a new one.
ACTIVE_STATUSES = ("pending", "confirmed")


def _parse_time_to_minutes(time_str: str):
    """Best-effort parse of a loosely-formatted time string ('2:00 PM',
    '14:00', '2pm', '2 pm') into minutes-since-midnight. Returns None if it
    can't be parsed — callers must handle that by falling back to an exact
    string match instead of silently skipping the conflict check."""
    cleaned = time_str.strip().upper().replace(".", "")
    formats = ["%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H:%M", "%H%M"]
    for fmt in formats:
        try:
            t = datetime.strptime(cleaned, fmt).time()
            return t.hour * 60 + t.minute
        except ValueError:
            continue
    return None


def _has_conflict(client_id: str, address: str, preferred_date: str, preferred_time: str) -> bool:
    existing = (
        supabase.table("showing_requests")
        .select("preferred_time, status")
        .eq("client_id", client_id)
        .eq("property_address", address)
        .eq("preferred_date", preferred_date)
        .in_("status", ACTIVE_STATUSES)
        .execute()
    )
    if not existing.data:
        return False

    new_minutes = _parse_time_to_minutes(preferred_time)

    for row in existing.data:
        existing_minutes = _parse_time_to_minutes(row["preferred_time"])
        if new_minutes is not None and existing_minutes is not None:
            if abs(new_minutes - existing_minutes) < SHOWING_BUFFER_MINUTES:
                return True
        else:
            # WHY: if either time string couldn't be parsed, fall back to an
            # exact string match rather than silently allowing a possible
            # double-booking through unchecked.
            if row["preferred_time"].strip().lower() == preferred_time.strip().lower():
                return True
    return False


def schedule_showing(client_id: str, session_id: str, property_address: str,
                     customer_name: str, customer_email: str, customer_phone: str,
                     preferred_date: str, preferred_time: str) -> dict:
    # 1. Email format
    if not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+$", customer_email.strip()):
        return {"error": "invalid_email",
                "message": "That email address doesn't look valid. Please double-check it."}

    # 2. Date must be a real, future (or today) calendar date: YYYY-MM-DD
    try:
        d = datetime.strptime(preferred_date.strip(), "%Y-%m-%d").date()
        if d < date.today():
            return {"error": "invalid_date",
                    "message": "That date is in the past. Please pick a future date."}
    except ValueError:
        return {"error": "invalid_date",
                "message": "I couldn't read that date. Please give a specific calendar date."}

    # 3. The property must exist for THIS client and still be available
    match = (
        supabase.table("properties")
        .select("address")
        .eq("client_id", client_id)
        .eq("status", "available")
        .ilike("address", f"%{property_address.strip()}%")
        .execute()
    )
    if not match.data:
        return {"error": "property_not_found",
                "message": "I can't find an available listing at that address."}

    resolved_address = match.data[0]["address"]

    # 4. Double-booking check — same property, same date, overlapping time
    if _has_conflict(client_id, resolved_address, preferred_date.strip(), preferred_time.strip()):
        return {
            "error": "time_conflict",
            "message": (
                "That time slot is already requested for this property. "
                "Please suggest a different time, and an agent will help "
                "coordinate."
            ),
        }

    result = (
        supabase.table("showing_requests")
        .insert({
            "client_id": client_id,
            "session_id": session_id,
            "property_address": resolved_address,
            "customer_name": customer_name,
            "customer_email": customer_email.strip(),
            "customer_phone": customer_phone,
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "status": "pending",
        })
        .execute()
    )
    row = result.data[0]

    client_resp = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
    webhook_url = client_resp.data[0].get("n8n_webhook_url") if client_resp.data else None
    if webhook_url:
        try:
            import httpx
            httpx.post(webhook_url, json=row, timeout=5)
        except Exception as e:
            print(f"[schedule_showing] Failed to notify n8n: {e}")

    return row