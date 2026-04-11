from dotenv import load_dotenv

from kume.infrastructure.config import Settings
from kume.infrastructure.container import Container
from kume.infrastructure.logging import setup_logging


def main() -> None:
    load_dotenv()
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    container = Container(settings)
    app = container.telegram_application()
    app.run_polling()


if __name__ == "__main__":
    main()
