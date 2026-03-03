"""
Meal pairing rules — which items complement each other.

Maps cuisine+sub_category to preferred sides, beverages, and desserts.
Used by the session generator to create realistic meal flows.
"""

from typing import Any, Dict

PAIRING_RULES: Dict[str, Dict[str, list]] = {
    "North Indian_Curry": {
        "preferred_sides": ["SIDE_001", "SIDE_002", "SIDE_005", "SIDE_006", "SIDE_007", "SIDE_010"],
        "preferred_beverages": ["BEV_003", "BEV_004", "BEV_005", "BEV_007"],
        "preferred_desserts": ["DES_001", "DES_002", "DES_004", "DES_006"],
    },
    "South Indian_Rice": {
        "preferred_sides": ["SIDE_005", "SIDE_010", "SIDE_009"],
        "preferred_beverages": ["BEV_003", "BEV_005", "BEV_002"],
        "preferred_desserts": ["DES_001", "DES_008", "DES_004"],
    },
    "Fast Food_Burger": {
        "preferred_sides": ["SIDE_003", "SIDE_004", "SIDE_012"],
        "preferred_beverages": ["BEV_001", "BEV_002", "BEV_009", "BEV_010"],
        "preferred_desserts": ["DES_003", "DES_007"],
    },
    "Italian_Pizza": {
        "preferred_sides": ["SIDE_012", "SIDE_003", "SIDE_008"],
        "preferred_beverages": ["BEV_001", "BEV_009", "BEV_012", "BEV_006"],
        "preferred_desserts": ["DES_005", "DES_003", "DES_007"],
    },
    "Chinese_Noodles": {
        "preferred_sides": ["SIDE_011", "SIDE_009", "SIDE_003"],
        "preferred_beverages": ["BEV_001", "BEV_009", "BEV_005"],
        "preferred_desserts": ["DES_003", "DES_004"],
    },
    "Healthy_Healthy": {
        "preferred_sides": ["SIDE_008", "SIDE_010"],
        "preferred_beverages": ["BEV_008", "BEV_011", "BEV_005"],
        "preferred_desserts": ["DES_004"],
    },
    "Street Food_Street Food": {
        "preferred_sides": ["SIDE_010", "SIDE_007", "SIDE_011"],
        "preferred_beverages": ["BEV_003", "BEV_005", "BEV_007", "BEV_002"],
        "preferred_desserts": ["DES_001", "DES_004", "DES_006"],
    },
}

DEFAULT_PAIRING: Dict[str, list] = {
    "preferred_sides": ["SIDE_001", "SIDE_003", "SIDE_005"],
    "preferred_beverages": ["BEV_001", "BEV_003", "BEV_005"],
    "preferred_desserts": ["DES_001", "DES_003"],
}
