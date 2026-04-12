from datetime import datetime

import pytest

from kume.domain.entities import Document, Goal, LabMarker, Restriction, User

NOW = datetime(2026, 1, 15, 10, 30, 0)


class TestUser:
    def test_creation(self) -> None:
        user = User(id="u-123", telegram_id=456)
        assert user.id == "u-123"
        assert user.telegram_id == 456

    def test_immutability(self) -> None:
        user = User(id="u-1", telegram_id=1)
        with pytest.raises(AttributeError):
            user.id = "u-2"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            user.telegram_id = 2  # type: ignore[misc]

    def test_equality(self) -> None:
        a = User(id="u-1", telegram_id=1)
        b = User(id="u-1", telegram_id=1)
        assert a == b

    def test_inequality(self) -> None:
        a = User(id="u-1", telegram_id=1)
        b = User(id="u-2", telegram_id=1)
        assert a != b


class TestGoal:
    def test_creation(self) -> None:
        goal = Goal(id="g-1", user_id="u-1", description="Lose 5kg", created_at=NOW)
        assert goal.id == "g-1"
        assert goal.user_id == "u-1"
        assert goal.description == "Lose 5kg"
        assert goal.created_at == NOW

    def test_immutability(self) -> None:
        goal = Goal(id="g-1", user_id="u-1", description="Lose 5kg", created_at=NOW)
        with pytest.raises(AttributeError):
            goal.id = "g-2"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            goal.description = "Gain muscle"  # type: ignore[misc]

    def test_field_types(self) -> None:
        goal = Goal(id="g-1", user_id="u-1", description="Lose 5kg", created_at=NOW)
        assert isinstance(goal.id, str)
        assert isinstance(goal.created_at, datetime)


class TestRestriction:
    def test_creation(self) -> None:
        r = Restriction(
            id="r-1",
            user_id="u-1",
            type="allergy",
            description="Peanuts",
            created_at=NOW,
        )
        assert r.id == "r-1"
        assert r.user_id == "u-1"
        assert r.type == "allergy"
        assert r.description == "Peanuts"
        assert r.created_at == NOW

    def test_immutability(self) -> None:
        r = Restriction(
            id="r-1",
            user_id="u-1",
            type="allergy",
            description="Peanuts",
            created_at=NOW,
        )
        with pytest.raises(AttributeError):
            r.type = "intolerance"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            r.description = "Gluten"  # type: ignore[misc]

    def test_field_types(self) -> None:
        r = Restriction(
            id="r-1",
            user_id="u-1",
            type="diet",
            description="Vegetarian",
            created_at=NOW,
        )
        assert isinstance(r.type, str)
        assert isinstance(r.created_at, datetime)


class TestDocument:
    def test_creation(self) -> None:
        doc = Document(
            id="d-1",
            user_id="u-1",
            type="lab_report",
            filename="blood_panel.pdf",
            summary="Complete blood panel results",
            ingested_at=NOW,
        )
        assert doc.id == "d-1"
        assert doc.user_id == "u-1"
        assert doc.type == "lab_report"
        assert doc.filename == "blood_panel.pdf"
        assert doc.summary == "Complete blood panel results"
        assert doc.ingested_at == NOW

    def test_immutability(self) -> None:
        doc = Document(
            id="d-1",
            user_id="u-1",
            type="lab_report",
            filename="blood_panel.pdf",
            summary="Results",
            ingested_at=NOW,
        )
        with pytest.raises(AttributeError):
            doc.filename = "other.pdf"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            doc.summary = "Changed"  # type: ignore[misc]

    def test_field_types(self) -> None:
        doc = Document(
            id="d-1",
            user_id="u-1",
            type="diet_plan",
            filename="plan.pdf",
            summary="Weekly plan",
            ingested_at=NOW,
        )
        assert isinstance(doc.filename, str)
        assert isinstance(doc.ingested_at, datetime)


class TestLabMarker:
    def test_creation(self) -> None:
        marker = LabMarker(
            id="lm-1",
            document_id="d-1",
            user_id="u-1",
            name="COLESTEROL TOTAL",
            value=185.0,
            unit="mg/dL",
            reference_range="< 200 mg/dL",
            date=NOW,
        )
        assert marker.id == "lm-1"
        assert marker.document_id == "d-1"
        assert marker.user_id == "u-1"
        assert marker.name == "COLESTEROL TOTAL"
        assert marker.value == 185.0
        assert marker.unit == "mg/dL"
        assert marker.reference_range == "< 200 mg/dL"
        assert marker.date == NOW

    def test_immutability(self) -> None:
        marker = LabMarker(
            id="lm-1",
            document_id="d-1",
            user_id="u-1",
            name="COLESTEROL TOTAL",
            value=185.0,
            unit="mg/dL",
            reference_range="< 200 mg/dL",
            date=NOW,
        )
        with pytest.raises(AttributeError):
            marker.value = 200.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            marker.name = "HDL"  # type: ignore[misc]

    def test_field_types(self) -> None:
        marker = LabMarker(
            id="lm-1",
            document_id="d-1",
            user_id="u-1",
            name="GLUCOSA",
            value=95.5,
            unit="mg/dL",
            reference_range="70-100 mg/dL",
            date=NOW,
        )
        assert isinstance(marker.value, float)
        assert isinstance(marker.unit, str)
        assert isinstance(marker.date, datetime)
