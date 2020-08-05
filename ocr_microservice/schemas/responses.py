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


class CVTemplateMatchResponse(Schema):
    """Response for CV Template match endpoint"""
    MATCHES = "matches"
    DOCUMENT_FILENAME = "document_filename"

    # This could be a nested schema, but that seems messier for the scope.
    matches = fields.List(fields.Dict)  # [{'template_text':str, 'bbox':[x,y,w,h], 'conf':#.#}]
    document_filename = fields.Str()


class PingResponse(Schema):
    """Reponse from ping endpoint."""

    status = fields.Str()
