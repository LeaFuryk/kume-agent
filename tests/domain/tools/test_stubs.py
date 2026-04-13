from kume.domain.tools.stubs import request_report


def test_request_report_returns_coming_soon() -> None:
    result = request_report()
    assert "coming soon" in result.lower()
    assert "nutrition reports" in result.lower()


def test_stubs_accept_arbitrary_kwargs_and_still_return_coming_soon() -> None:
    assert "coming soon" in request_report(period="weekly", format="pdf").lower()
