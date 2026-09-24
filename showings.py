# showings.py
# WHAT: Logs a property showing REQUEST (never a confirmation) and notifies
# the client's n8n webhook (if configured).
# WHY: Hard validation lives in CODE, not just the prompt — the model can
# forget rules, code can't.
import os
import re
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


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

    result = (
        supabase.table("showing_requests")
        .insert({
            "client_id": client_id,
            "session_id": session_id,
            "property_address": match.data[0]["address"],
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









# # showings.py
# # WHAT: Logs a property showing request and notifies the client's n8n
# # webhook (if configured) — replaces confirmed_orders.py for real estate.
# import os
# from dotenv import load_dotenv
# from supabase import create_client, Client

# load_dotenv()
# supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# def schedule_showing(client_id: str, session_id: str, property_address: str,
#                       customer_name: str, customer_email: str, customer_phone: str,
#                       preferred_date: str, preferred_time: str) -> dict:
#     result = (
#         supabase.table("showing_requests")
#         .insert({
#             "client_id": client_id,
#             "session_id": session_id,
#             "property_address": property_address,
#             "customer_name": customer_name,
#             "customer_email": customer_email,
#             "customer_phone": customer_phone,
#             "preferred_date": preferred_date,
#             "preferred_time": preferred_time,
#             "status": "pending",
#         })
#         .execute()
#     )
#     order_row = result.data[0]

#     client_resp = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
#     webhook_url = client_resp.data[0].get("n8n_webhook_url") if client_resp.data else None
#     if webhook_url:
#         try:
#             import httpx
#             httpx.post(webhook_url, json=order_row, timeout=5)
#         except Exception as e:
#             print(f"[schedule_showing] Failed to notify n8n: {e}")

#     return order_row