from kume.domain.tools.stubs import ingest_context, log_meal, request_report


def test_ingest_context_returns_coming_soon() -> None:
    result = ingest_context()
    assert "coming soon" in result.lower()
    assert "health documents" in result.lower()


def test_log_meal_returns_coming_soon() -> None:
    result = log_meal()
    assert "coming soon" in result.lower()
    assert "track your meals" in result.lower()


def test_request_report_returns_coming_soon() -> None:
    result = request_report()
    assert "coming soon" in result.lower()
    assert "nutrition reports" in result.lower()


def test_stubs_accept_arbitrary_kwargs_and_still_return_coming_soon() -> None:
    assert "coming soon" in ingest_context(file="test.pdf", user_id=123).lower()
    assert "coming soon" in log_meal(food="pasta", calories=400).lower()
    assert "coming soon" in request_report(period="weekly", format="pdf").lower()
