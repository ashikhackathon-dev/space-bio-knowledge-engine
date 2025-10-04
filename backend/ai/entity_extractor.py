from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ENTITY_TYPES = [
    "Organism",
    "Tissue",
    "Assay",
    "Mission",
    "Hazard",
    "Countermeasure",
    "Gene",
    "Protein",
    "Pathway",
]


@dataclass
class ExtractedEntity:
    ent_type: str
    text: str
    normalized: Optional[str]
    start: Optional[int]
    end: Optional[int]


@dataclass
class ExtractedRelation:
    source_text: str
    target_text: str
    rel_type: str


def simple_rule_entities(text: str) -> List[ExtractedEntity]:
    # Placeholder heuristic rules; in production, replace with model-based IE
    entities: List[ExtractedEntity] = []
    tokens = text.split()
    for idx, tok in enumerate(tokens):
        if tok.lower() in {"mouse", "mice", "murine", "rat", "zebrafish"}:
            entities.append(ExtractedEntity("Organism", tok, tok.lower(), None, None))
        if tok.lower() in {"rna-seq", "rna", "proteomics", "metabolomics"}:
            entities.append(ExtractedEntity("Assay", tok, tok.lower(), None, None))
        if tok.lower() in {"iss", "apollo", "spacex", "crs"}:
            entities.append(ExtractedEntity("Mission", tok, tok.upper(), None, None))
    return entities


def simple_relations(entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
    # Dummy relation: link any Organism to any Assay with STUDIES
    rels: List[ExtractedRelation] = []
    organisms = [e for e in entities if e.ent_type == "Organism"]
    assays = [e for e in entities if e.ent_type == "Assay"]
    for o in organisms:
        for a in assays:
            rels.append(ExtractedRelation(o.text, a.text, "STUDIES"))
    return rels



