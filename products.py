# products.py
# WHAT: Product search, now backed by Supabase instead of a hardcoded list,
# and ALWAYS scoped to a single client_id.
# WHY: This is the same function signature change discussed back in Phase 2 —
# we said the mock data would eventually be swapped for a real database
# without touching the agent code. This is that swap.

import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)


def search_products(
    client_id: str,
    category: str | None = None,
    max_price: float | None = None,
    in_stock_only: bool = True,
) -> list[dict]:
    # WHY: client_id filter is ALWAYS applied first and is never optional —
    # this is the actual security boundary of multi-tenancy. Every other
    # filter is optional; this one never is.
    query = supabase.table("products").select("*").eq("client_id", client_id)

    if category:
        query = query.ilike("category", category)
    if max_price is not None:
        query = query.lte("price", max_price)
    if in_stock_only:
        query = query.eq("in_stock", True)

    response = query.execute()
    return response.data

def find_solution(client_id: str, concern: str) -> list[dict]:
    """Search for products addressing a customer concern. Matches on
    INDIVIDUAL keywords, not the whole phrase — 'acne and oily skin' now
    correctly matches products tagged 'acne' OR 'oily skin' even though
    that exact combined phrase never appears verbatim in the data."""
    # WHY: strip common filler words so "and", "my", "skin is so" don't
    # dilute the match — keep the meaningful keywords only.
    stopwords = {"and", "my", "is", "so", "the", "a", "i", "have", "has", "it", "in"}
    words = [w.strip(",.!?") for w in concern.lower().split()]
    keywords = [w for w in words if w and w not in stopwords and len(w) > 2]

    if not keywords:
        return []

    # WHY: build one OR condition per keyword, checked against BOTH fields
    # — matches if ANY meaningful word from the customer's phrasing appears
    # anywhere in the tags or description.
    conditions = []
    for kw in keywords:
        conditions.append(f"concerns_solved.ilike.%{kw}%")
        conditions.append(f"description.ilike.%{kw}%")

    response = (
        supabase.table("products")
        .select("*")
        .eq("client_id", client_id)
        .eq("in_stock", True)
        .or_(",".join(conditions))
        .execute()
    )
    return response.data