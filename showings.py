# showings.py
# WHAT: Logs a property showing request and notifies the client's n8n
# webhook (if configured) — replaces confirmed_orders.py for real estate.
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def schedule_showing(client_id: str, session_id: str, property_address: str,
                      customer_name: str, customer_email: str, customer_phone: str,
                      preferred_date: str, preferred_time: str) -> dict:
    result = (
        supabase.table("showing_requests")
        .insert({
            "client_id": client_id,
            "session_id": session_id,
            "property_address": property_address,
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_phone": customer_phone,
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "status": "pending",
        })
        .execute()
    )
    order_row = result.data[0]

    client_resp = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
    webhook_url = client_resp.data[0].get("n8n_webhook_url") if client_resp.data else None
    if webhook_url:
        try:
            import httpx
            httpx.post(webhook_url, json=order_row, timeout=5)
        except Exception as e:
            print(f"[schedule_showing] Failed to notify n8n: {e}")

    return order_row