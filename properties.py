# properties.py
# WHAT: Property search and need-based matching, always scoped to one client_id.
# WHY (Fair Housing): listing text is passed through _clean() before the model
# ever sees it, so phrases about neighborhood character, safety, schools, or
# who a home "suits" can't be repeated to customers (steering risk).
# This is a safety net — listing descriptions should still be written
# objectively.
import os
import re
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

MAX_RESULTS = 5

_STEERING = re.compile(
    r"\b(famil(?:y|ies)(?![\s-]room)|safe|safest|safety|crime|exclusive|prestigious|"
    r"desirable|traditional|up-and-coming|good schools?|great schools?|"
    r"school district|quiet neighborhood|neighbou?rhood)\b",
    re.I,
)


def _clean(p: dict) -> dict:
    """Drop sentences/features describing neighborhood character or who a home suits."""
    if p.get("description"):
        sentences = re.split(r"(?<=[.!?])\s+", p["description"])
        p["description"] = " ".join(s for s in sentences if not _STEERING.search(s))
    if p.get("features"):
        p["features"] = ", ".join(
            f.strip() for f in p["features"].split(",") if not _STEERING.search(f)
        )
    return p


def search_properties(client_id: str, listing_type: str = None, city: str = None,
                      min_price: float = None, max_price: float = None,
                      bedrooms: int = None) -> list[dict]:
    query = (
        supabase.table("properties")
        .select("*")
        .eq("client_id", client_id)
        .eq("status", "available")
    )
    if listing_type:
        query = query.eq("listing_type", listing_type)
    if city:
        query = query.ilike("city", f"%{city}%")
    if min_price is not None:
        query = query.gte("price", min_price)
    if max_price is not None:
        query = query.lte("price", max_price)
    if bedrooms is not None:
        query = query.gte("bedrooms", bedrooms)
    return [_clean(p) for p in query.limit(MAX_RESULTS).execute().data]


def find_property_match(client_id: str, needs: str) -> list[dict]:
    stopwords = {"and", "my", "is", "so", "the", "a", "i", "have", "need", "want", "for", "with", "near"}
    words = [w.strip(",.!?") for w in needs.lower().split()]
    keywords = [w for w in words if w and w not in stopwords and len(w) > 2]
    if not keywords:
        return []
    conditions = []
    for kw in keywords:
        conditions.append(f"features.ilike.%{kw}%")
        conditions.append(f"description.ilike.%{kw}%")
        conditions.append(f"city.ilike.%{kw}%")
    data = (
        supabase.table("properties")
        .select("*")
        .eq("client_id", client_id)
        .eq("status", "available")
        .or_(",".join(conditions))
        .limit(MAX_RESULTS)
        .execute()
        .data
    )
    return [_clean(p) for p in data]








# # properties.py
# # WHAT: Property search and need-based matching — replaces products.py
# # for the real estate domain.
# import os
# from dotenv import load_dotenv
# from supabase import create_client, Client

# load_dotenv()
# supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


# def search_properties(client_id: str, listing_type: str = None, city: str = None,
#                        min_price: float = None, max_price: float = None,
#                        bedrooms: int = None) -> list[dict]:
#     query = supabase.table("properties").select("*").eq("client_id", client_id).eq("status", "available")
#     if listing_type:
#         query = query.eq("listing_type", listing_type)
#     if city:
#         query = query.ilike("city", f"%{city}%")
#     if min_price is not None:
#         query = query.gte("price", min_price)
#     if max_price is not None:
#         query = query.lte("price", max_price)
#     if bedrooms is not None:
#         query = query.gte("bedrooms", bedrooms)
#     return query.execute().data


# def find_property_match(client_id: str, needs: str) -> list[dict]:
#     stopwords = {"and", "my", "is", "so", "the", "a", "i", "have", "need", "want", "for", "with", "near"}
#     words = [w.strip(",.!?") for w in needs.lower().split()]
#     keywords = [w for w in words if w and w not in stopwords and len(w) > 2]
#     if not keywords:
#         return []
#     conditions = []
#     for kw in keywords:
#         conditions.append(f"features.ilike.%{kw}%")
#         conditions.append(f"description.ilike.%{kw}%")
#         conditions.append(f"city.ilike.%{kw}%")
#     return (
#         supabase.table("properties")
#         .select("*")
#         .eq("client_id", client_id)
#         .eq("status", "available")
#         .or_(",".join(conditions))
#         .execute()
#         .data
#     )