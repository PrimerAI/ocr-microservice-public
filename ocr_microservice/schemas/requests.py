from flask_marshmallow.fields import fields

from primer_micro_utils.namespace import Schema


class OCRRequest(Schema):
    """Parameter definition for OCR requests"""

    FILE_TYPE = "fileType"
    LANGUAGE = "language"
    DETECT_ORIENTATION = "detectOrientation"
    BODY = "body"

    language = fields.Str(
        attribute=FILE_TYPE,
        description="File type as a ",
        missing="en",
        required=False,
    )

    language = fields.Str(
        attribute=LANGUAGE,
        description="The BCP-47 language code of the text to be detected. The default is 'en'.",
        missing="en",
        required=False,
    )
    detect_orientation = fields.Boolean(
        attribute=DETECT_ORIENTATION,
        required=False,
        missing=False,
        description="Whether detect the text orientation in the image.",
    )
    body = fields.Raw(
        attribute=BODY,
        required=True,
        description="File data to be processed",
    )


class CVTemplateMatchRequest(Schema):
    """Parameter definition for doing template matching on a document."""
    DOCUMENT = "document"
    TEMPLATE = "template"

    document = fields.Raw(
        attribute=DOCUMENT,
        required=True,
        description="A PDF or image to be used as the target for multi-resolution matching.",
    )

    template = fields.Raw(
        attribute=TEMPLATE,
        required=True,
        description="An image to be used as a template for searching inside the document."
    )


class CVTextMatchRequest(Schema):
    """Similar to CVTemplateRequest, but takes text to be matched and rasterizes it."""
    DOCUMENT = "document"
    TEXT = "text"
    THRESHOLD = "threshold"

    document = fields.Raw(
        attribute=DOCUMENT,
        required=True,
        description="A PDF or image to be used as the target for multi-resolution matching.",
    )

    text = fields.Str(
        attribute=TEXT,
        required=True,
        description="A string to be rasterized and used to search the document."
    )

    threshold = fields.Float(
        attribute=THRESHOLD,
        required=False,
        missing=0.9,
        description="Matches which fall below this value will be culled and not returned."
    )
