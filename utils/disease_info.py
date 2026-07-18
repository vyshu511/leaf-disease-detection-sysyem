"""
Disease Information Database
-----------------------------
Maps a predicted class name (as produced by class_names.npy, typically in the
PlantVillage-style format "Plant___Disease_Name") to structured agronomic
information: symptoms, causes, prevention, organic remedies, chemical
treatments, and farming tips.

The DB is intentionally split into two layers:

1. EXACT_MATCH  - keyed on the exact class name string used by your dataset.
                  Fill this in with your real 12 class names for the most
                  accurate, disease-specific guidance.
2. KEYWORD_RULES - a fallback that pattern-matches on words inside the class
                  name (e.g. "blight", "rust", "healthy") so the app never
                  shows a blank panel even for classes you haven't manually
                  curated yet.

To adapt this to your exact dataset:
    >>> import numpy as np
    >>> print(np.load("model/class_names.npy", allow_pickle=True))
Then copy each class name into EXACT_MATCH below and fill in the fields.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class DiseaseInfo:
    display_name: str
    is_healthy: bool = False
    symptoms: List[str] = field(default_factory=list)
    causes: List[str] = field(default_factory=list)
    prevention: List[str] = field(default_factory=list)
    organic_remedies: List[str] = field(default_factory=list)
    chemical_treatments: List[str] = field(default_factory=list)
    farming_tips: List[str] = field(default_factory=list)
    severity: str = "moderate"  # low | moderate | high


# ---------------------------------------------------------------------------
# 1. EXACT MATCHES — replace / extend with your real class names
# ---------------------------------------------------------------------------
EXACT_MATCH = {
    "Tomato___Healthy": DiseaseInfo(
        display_name="Healthy Tomato Leaf",
        is_healthy=True,
        symptoms=["Uniform green color", "No spots, wilting, or curling"],
        causes=[],
        prevention=[
            "Maintain a consistent watering schedule",
            "Ensure 6+ hours of sunlight daily",
            "Rotate crops each season",
        ],
        organic_remedies=[],
        chemical_treatments=[],
        farming_tips=[
            "Continue regular monitoring every 5-7 days",
            "Apply balanced NPK fertilizer during fruiting stage",
        ],
        severity="low",
    ),
    "Tomato___Early_Blight": DiseaseInfo(
        display_name="Tomato Early Blight",
        symptoms=[
            "Dark concentric 'target-board' rings on older leaves",
            "Yellowing around lesions",
            "Leaf drop starting from the bottom of the plant",
        ],
        causes=[
            "Fungus Alternaria solani",
            "Warm, humid weather with periods of leaf wetness",
            "Overhead irrigation splashing spores onto leaves",
        ],
        prevention=[
            "Water at the base of the plant, not overhead",
            "Space plants for good air circulation",
            "Remove and destroy infected lower leaves promptly",
            "Rotate with non-solanaceous crops for 2 years",
        ],
        organic_remedies=[
            "Neem oil spray (2%) every 7-10 days",
            "Baking soda solution (1 tbsp/gallon water) as a foliar spray",
            "Copper-based organic fungicide (OMRI listed)",
            "Compost tea to boost plant immunity",
        ],
        chemical_treatments=[
            "Chlorothalonil-based fungicide per label rate",
            "Mancozeb spray at first sign of symptoms",
            "Azoxystrobin for severe outbreaks (follow local regulations)",
        ],
        farming_tips=[
            "Mulch soil to prevent spore splash-back",
            "Stake or cage plants to keep foliage off the ground",
        ],
        severity="high",
    ),
}


# ---------------------------------------------------------------------------
# 2. KEYWORD FALLBACK — used when a class isn't in EXACT_MATCH above
# ---------------------------------------------------------------------------
KEYWORD_RULES = [
    (["healthy"], DiseaseInfo(
        display_name="Healthy Leaf",
        is_healthy=True,
        symptoms=["Even leaf color", "No visible lesions or discoloration"],
        prevention=["Keep up current watering and fertilizing routine",
                    "Inspect weekly for early signs of stress"],
        farming_tips=["Healthy leaves are the best time to apply preventive, "
                       "non-chemical care such as compost top-dressing"],
        severity="low",
    )),
    (["blight"], DiseaseInfo(
        display_name="Blight",
        symptoms=["Brown/black necrotic patches", "Rapid leaf collapse",
                   "Concentric rings on lesions (early blight) or water-soaked "
                   "patches (late blight)"],
        causes=["Fungal or oomycete pathogen favored by damp, humid conditions"],
        prevention=["Avoid overhead watering", "Improve air circulation",
                     "Remove infected debris at season end"],
        organic_remedies=["Copper-based fungicide", "Neem oil spray",
                            "Baking soda + horticultural oil spray"],
        chemical_treatments=["Chlorothalonil", "Mancozeb", "Copper oxychloride"],
        farming_tips=["Rotate crops for at least 2 seasons",
                       "Avoid working in fields when foliage is wet"],
        severity="high",
    )),
    (["rust"], DiseaseInfo(
        display_name="Rust",
        symptoms=["Orange/rust-colored pustules on leaf undersides",
                   "Yellow spots on upper leaf surface"],
        causes=["Fungal spores (Puccinia spp.) spread by wind and rain"],
        prevention=["Plant rust-resistant varieties", "Avoid dense planting",
                     "Remove volunteer plants that host the fungus"],
        organic_remedies=["Sulfur dust or spray", "Neem oil"],
        chemical_treatments=["Propiconazole", "Tebuconazole-based fungicide"],
        farming_tips=["Destroy crop residue after harvest to reduce spores"],
        severity="moderate",
    )),
    (["mildew"], DiseaseInfo(
        display_name="Powdery / Downy Mildew",
        symptoms=["White or grayish powdery coating on leaves",
                   "Leaf curling and yellowing"],
        causes=["Fungal pathogens thriving in humid, poorly ventilated conditions"],
        prevention=["Increase plant spacing", "Prune for airflow",
                     "Avoid excess nitrogen fertilizer"],
        organic_remedies=["Milk spray (1:9 milk:water)", "Sulfur spray",
                            "Potassium bicarbonate solution"],
        chemical_treatments=["Myclobutanil", "Triadimefon"],
        farming_tips=["Water early in the day so foliage dries before evening"],
        severity="moderate",
    )),
    (["bacterial", "spot"], DiseaseInfo(
        display_name="Bacterial Spot",
        symptoms=["Small water-soaked spots with yellow halos",
                   "Spots turn dark brown and may merge"],
        causes=["Xanthomonas bacteria spread by splashing water and contaminated tools"],
        prevention=["Use certified disease-free seed/seedlings",
                     "Disinfect tools between plants", "Avoid working wet fields"],
        organic_remedies=["Copper-based bactericide", "Compost tea foliar spray"],
        chemical_treatments=["Copper hydroxide + Mancozeb combination sprays"],
        farming_tips=["Practice at least a 2-year crop rotation"],
        severity="moderate",
    )),
    (["virus", "mosaic", "curl"], DiseaseInfo(
        display_name="Viral Infection",
        symptoms=["Mottled yellow-green mosaic pattern", "Leaf curling and distortion",
                   "Stunted plant growth"],
        causes=["Plant virus, often transmitted by aphids or whiteflies"],
        prevention=["Control insect vectors (aphids/whiteflies)",
                     "Remove and destroy infected plants immediately",
                     "Use virus-resistant cultivars"],
        organic_remedies=["Neem oil to control insect vectors",
                            "Yellow sticky traps for whiteflies"],
        chemical_treatments=["Insecticidal control of vector insects "
                               "(no direct chemical cure for the virus itself)"],
        farming_tips=["There is no cure once infected — focus entirely on prevention "
                       "and vector control"],
        severity="high",
    )),
    (["mold"], DiseaseInfo(
        display_name="Leaf Mold",
        symptoms=["Pale green/yellow spots on upper leaf surface",
                   "Olive-green to grayish-purple fuzzy growth underneath"],
        causes=["Fungus favored by high humidity (>85%) and poor ventilation"],
        prevention=["Improve greenhouse/field ventilation", "Reduce humidity",
                     "Avoid overhead irrigation"],
        organic_remedies=["Neem oil", "Copper-based organic fungicide"],
        chemical_treatments=["Chlorothalonil", "Copper oxychloride"],
        farming_tips=["Prune lower leaves to improve air movement"],
        severity="moderate",
    )),
]

_GENERIC_FALLBACK = DiseaseInfo(
    display_name="Unrecognized / Unmapped Class",
    symptoms=["Refer to a local agricultural extension for exact diagnosis"],
    prevention=["Maintain good field hygiene", "Rotate crops",
                 "Avoid overhead watering"],
    organic_remedies=["Neem oil spray is a safe general-purpose first response"],
    chemical_treatments=["Consult a local agronomist before applying chemicals"],
    farming_tips=["Take a clearer, well-lit photo for a more confident diagnosis"],
    severity="unknown",
)


def get_disease_info(class_name: str) -> DiseaseInfo:
    """Look up structured info for a predicted class name."""
    if class_name in EXACT_MATCH:
        return EXACT_MATCH[class_name]

    lowered = class_name.lower()
    for keywords, info in KEYWORD_RULES:
        if any(k in lowered for k in keywords):
            return info

    return _GENERIC_FALLBACK


def to_dict(info: DiseaseInfo) -> dict:
    """Serialize a DiseaseInfo to a plain dict for JSON / template use."""
    return {
        "display_name": info.display_name,
        "is_healthy": info.is_healthy,
        "symptoms": info.symptoms,
        "causes": info.causes,
        "prevention": info.prevention,
        "organic_remedies": info.organic_remedies,
        "chemical_treatments": info.chemical_treatments,
        "farming_tips": info.farming_tips,
        "severity": info.severity,
    }
