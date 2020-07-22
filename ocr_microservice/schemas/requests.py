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
