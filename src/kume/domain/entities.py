from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class User:
    """A Telegram user of the Kume nutrition bot."""

    id: str
    telegram_id: int
    name: str | None = None
    language: str = "en"
    timezone: str = "UTC"


@dataclass(frozen=True)
class Goal:
    """A nutrition or health goal set by the user."""

    id: str
    user_id: str
    description: str
    created_at: datetime
    completed_at: datetime | None = None


@dataclass(frozen=True)
class Restriction:
    """A dietary restriction (allergy, intolerance, or diet preference)."""

    id: str
    user_id: str
    type: str  # "allergy", "intolerance", "diet"
    description: str
    created_at: datetime
    completed_at: datetime | None = None


@dataclass(frozen=True)
class Document:
    """An ingested document (lab report, diet plan, or medical report)."""

    id: str
    user_id: str
    type: str  # "lab_report", "diet_plan", "medical_report"
    filename: str
    summary: str
    ingested_at: datetime


@dataclass(frozen=True)
class LabMarker:
    """A single lab marker extracted from a document."""

    id: str
    document_id: str
    user_id: str
    name: str  # "COLESTEROL TOTAL"
    value: float
    unit: str  # "mg/dL"
    reference_range: str  # "< 200 mg/dL"
    date: datetime


@dataclass(frozen=True)
class Meal:
    """A logged meal with nutritional breakdown."""

    id: str
    user_id: str
    description: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sodium_mg: float
    sugar_g: float
    saturated_fat_g: float
    cholesterol_mg: float
    confidence: float  # 0.0–1.0
    image_present: bool
    logged_at: datetime


@dataclass(frozen=True)
class NutritionEstimate:
    """Intermediate output from vision analysis, before becoming a Meal."""

    description: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float
    sodium_mg: float
    sugar_g: float
    saturated_fat_g: float
    cholesterol_mg: float
    confidence: float
