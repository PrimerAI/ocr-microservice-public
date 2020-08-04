"""Export all."""

from ocr_microservice.routes.ocr import ocr
from ocr_microservice.routes.ping import ping

__all__ = [
    "ping",
    "cv",
    "ocr",
]
