"""Session simulation — generates realistic user cart sessions."""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from swaadstack.config import data_config
from swaadstack.data.menu_items import MENU_ITEMS
from swaadstack.data.pairing_rules import PAIRING_RULES, DEFAULT_PAIRING
from swaadstack.utils.encoding import get_mealtime_label
from swaadstack.utils.helpers import console


def get_pairing_key(item: Dict[str, Any]) -> str:
    """Get the pairing rule key for a menu item."""
    return f"{item['cuisine']}_{item['sub_category']}"


def simulate_sessions(
    items: List[Dict[str, Any]],
    num_sessions: int = 5000,
    num_users: int = 500,
    geohashes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate realistic user sessions following meal logic.

    Session flow: Main -> Side -> Beverage -> Dessert
    with conditional parameterization on user profile, time, and location.
    """
    if geohashes is None:
        geohashes = data_config.geohashes

    items_by_category: Dict[str, List] = {}
    items_by_id: Dict[str, Dict] = {}
    for item in items:
        cat = item["category"]
        if cat not in items_by_category:
            items_by_category[cat] = []
        items_by_category[cat].append(item)
        items_by_id[item["item_id"]] = item

    category_flow = ["Main", "Side", "Beverage", "Dessert"]

    user_profiles = {}
    for uid in range(num_users):
        user_profiles[f"user_{uid:04d}"] = {
            "budget": random.choice(["low", "medium", "high"]),
            "cuisine_affinity": random.choice(
                ["North Indian", "South Indian", "Fast Food", "Italian", "Chinese", "Mixed"]
            ),
            "dietary": random.choice(["Vegetarian", "Non-Vegetarian", "Mixed"]),
            "preferred_geohash": random.choice(geohashes),
        }

    sessions = []
    base_time = datetime(2025, 1, 1)

    console.print(f"[cyan]Simulating {num_sessions} sessions for {num_users} users...[/cyan]")

    for session_idx in tqdm(range(num_sessions), desc="Generating sessions"):
        user_id = random.choice(list(user_profiles.keys()))
        profile = user_profiles[user_id]

        day_offset = random.randint(0, 42)
        hour = _sample_meal_hour()
        minute = random.randint(0, 59)
        timestamp = base_time + timedelta(days=day_offset, hours=hour, minutes=minute)

        geohash = profile["preferred_geohash"] if random.random() < 0.7 else random.choice(geohashes)
        mealtime = get_mealtime_label(hour)

        session_length = _sample_session_length()

        cart_sequence = []

        main_item = _select_item(items_by_category["Main"], profile, mealtime, geohash)
        cart_sequence.append(main_item)

        pairing_key = get_pairing_key(main_item)
        pairing = PAIRING_RULES.get(pairing_key, DEFAULT_PAIRING)

        remaining_categories = category_flow[1:]

        for step in range(1, session_length):
            if step <= len(remaining_categories):
                next_cat = remaining_categories[step - 1]
            else:
                next_cat = random.choice(category_flow)

            if next_cat == "Side" and random.random() < 0.7:
                preferred = pairing.get("preferred_sides", [])
                candidates = [items_by_id[pid] for pid in preferred if pid in items_by_id]
                if not candidates:
                    candidates = items_by_category.get("Side", [])
            elif next_cat == "Beverage" and random.random() < 0.7:
                preferred = pairing.get("preferred_beverages", [])
                candidates = [items_by_id[pid] for pid in preferred if pid in items_by_id]
                if not candidates:
                    candidates = items_by_category.get("Beverage", [])
            elif next_cat == "Dessert" and random.random() < 0.6:
                preferred = pairing.get("preferred_desserts", [])
                candidates = [items_by_id[pid] for pid in preferred if pid in items_by_id]
                if not candidates:
                    candidates = items_by_category.get("Dessert", [])
            else:
                candidates = items_by_category.get(next_cat, items_by_category["Side"])

            next_item = _select_item(candidates, profile, mealtime, geohash)

            while next_item["item_id"] in [i["item_id"] for i in cart_sequence] and len(candidates) > 1:
                candidates = [c for c in candidates if c["item_id"] != next_item["item_id"]]
                next_item = _select_item(candidates, profile, mealtime, geohash)

            cart_sequence.append(next_item)

        for i in range(1, len(cart_sequence)):
            context_ids = [item["item_id"] for item in cart_sequence[:i]]
            target_id = cart_sequence[i]["item_id"]

            sessions.append({
                "user_id": user_id,
                "session_id": f"sess_{session_idx:06d}",
                "sequence_item_ids": "|".join(context_ids),
                "target_item_id": target_id,
                "target_category": cart_sequence[i]["category"],
                "cart_size": i,
                "timestamp": timestamp.isoformat(),
                "hour": hour,
                "day_of_week": timestamp.weekday(),
                "mealtime": mealtime,
                "geohash": geohash,
            })

    df = pd.DataFrame(sessions)
    console.print(f"[green]✓ Generated {len(df)} training samples from {num_sessions} sessions[/green]")
    return df


# ── Private helpers ──

def _sample_meal_hour() -> int:
    """Sample hour with realistic mealtime distribution."""
    probabilities = [
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.06, 0.06, 0.04, 0.06,
        0.10, 0.08, 0.05, 0.03, 0.03, 0.04, 0.06, 0.08, 0.08, 0.06, 0.04, 0.03,
    ]
    return random.choices(list(range(24)), weights=probabilities, k=1)[0]


def _sample_session_length() -> int:
    """Sample cart size — most users add 2–3 items."""
    return random.choices([2, 3, 4, 5], weights=[0.25, 0.40, 0.25, 0.10], k=1)[0]


def _select_item(
    candidates: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    mealtime: str,
    geohash: str,
) -> Dict[str, Any]:
    """Select item based on user profile and context (conditional parameterization)."""
    if not candidates:
        return random.choice(MENU_ITEMS)

    weighted_candidates = []
    for item in candidates:
        weight = 1.0

        if user_profile["budget"] == "low" and item["price"] > 300:
            weight *= 0.3
        elif user_profile["budget"] == "high" and item["price"] < 100:
            weight *= 0.5

        if user_profile["cuisine_affinity"] != "Mixed":
            if item["cuisine"] == user_profile["cuisine_affinity"]:
                weight *= 2.0

        if user_profile["dietary"] == "Vegetarian":
            if "Non-Vegetarian" in item["dietary"]:
                weight *= 0.05
            elif "Vegetarian" in item["dietary"]:
                weight *= 1.5

        if mealtime == "breakfast" and item.get("sub_category") in ["South Indian", "Bread"]:
            weight *= 1.5
        elif mealtime == "late_night" and item.get("sub_category") in ["Ice Cream", "Snack", "Milkshake"]:
            weight *= 2.0

        weighted_candidates.append((item, weight))

    items_list, weights = zip(*weighted_candidates)
    return random.choices(items_list, weights=weights, k=1)[0]
