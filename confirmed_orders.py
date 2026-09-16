# confirmed_orders.py
# WHAT: Logs a confirmed order to Supabase and notifies the client's n8n
# webhook (if configured), which handles however THAT business wants to
# be notified (Slack, email, WhatsApp — n8n's job, not ours).
import os
import httpx
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


def confirm_order(
    client_id: str,
    session_id: str,
    product_name: str,
    quantity: int,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    shipping_address: str,
    country: str,
) -> dict:
    # WHY: hard validation in CODE, not just a prompt instruction — this is
    # too consequential (a real unfulfillable order) to trust the model
    # alone to remember. This is the same philosophy as Phase 4's order
    # security check: never trust the model to self-enforce a hard rule.
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", customer_email):
        return {"error": "invalid_email", "message": "That email address doesn't look valid — please double check it."}

    client_resp = supabase.table("clients").select("shipping_countries").eq("client_id", client_id).execute()
    allowed = client_resp.data[0]["shipping_countries"] if client_resp.data else "US"
    allowed_list = [c.strip().lower() for c in allowed.split(",")]

    if country.strip().lower() not in allowed_list and country.strip().lower() not in ("us", "usa", "united states"):
        return {
            "error": "not_serviceable",
            "message": f"Sorry, we currently only ship within: {allowed}. We can't ship to {country} yet.",
        }

    result = (
        supabase.table("confirmed_orders")
        .insert({
            "client_id": client_id, "session_id": session_id, "product_name": product_name,
            "quantity": quantity, "customer_name": customer_name, "customer_email": customer_email,
            "customer_phone": customer_phone, "shipping_address": f"{shipping_address}, {country}",
            "status": "pending",
        })
        .execute()
    )
    order_row = result.data[0]

    client_resp2 = supabase.table("clients").select("n8n_webhook_url").eq("client_id", client_id).execute()
    webhook_url = client_resp2.data[0].get("n8n_webhook_url") if client_resp2.data else None
    if webhook_url:
        try:
            httpx.post(webhook_url, json=order_row, timeout=5)
        except Exception as e:
            print(f"[confirm_order] Failed to notify n8n: {e}")

    return order_row