from flask_marshmallow.fields import fields

from primer_micro_utils.namespace import Schema


class OCRResponse(Schema):
    """Response for OCR endpoint"""

    TEXT = "text"
    FILE_NAME = "filename"
    CONTENT_TYPE = "content_type"
    CONTENT_LENGTH = "content_length"
    MIMETYPE = "mimetype"

    text = fields.Str()
    filename = fields.Str()
    content_type = fields.Str()
    content_length = fields.Str()
    mimetype = fields.Str()


class PingResponse(Schema):
    """Reponse from ping endpoint."""

    status = fields.Str()
