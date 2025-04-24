import logging, sys, structlog


def configure_logging(level: str = "INFO"):
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=numeric, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
    )
