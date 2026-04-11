__all__ = ["TelegramMessagingAdapter"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "TelegramMessagingAdapter":
        from kume.adapters.output.telegram_messaging import TelegramMessagingAdapter

        return TelegramMessagingAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
