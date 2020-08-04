"""
An endpoint for doing rudimentary document recognition with simple CV techniques.
"""

from flask_restplus import Resource
from werkzeug.datastructures import FileStorage

from ocr_microservice.common import match_template_in_image
from ocr_microservice.schemas import (
    CVTextMatchRequest, CVTemplateMatchRequest, CVTemplateMatchResponse
)
from primer_micro_utils.namespace import Namespace

cv = Namespace("cv", description="Computer Vision Methods")

# TODO(jc): This is copy-pasted from the original.  Can we make this part of the request object?
doc_upload_parser = cv.parser()
doc_upload_parser.add_argument(
    'doc_file', location='files', type=FileStorage, required=True,
)

template_upload_parser = cv.parser()
template_upload_parser.add_argument(
    'template_file', location='files', type=FileStorage, required=True
)


@cv.route("/template-match", strict_slashes=False, methods=["POST"])
class CVHandler(Resource):
    @cv.response(CVTemplateMatchRequest())
    @cv.expect(doc_upload_parser)
    @cv.expect(template_upload_parser)
    @cv.response(CVTemplateMatchResponse())
    def post(self):
        """Read file and template.  Try to match one in the other."""
        args = doc_upload_parser.parse_args()
        uploaded_file = args['file']  # FileStorage instance
        args = template_upload_parser.parse_args()
        template_file = args['file']
        matches = match_template_in_image(uploaded_file, template_file)
        return {

        }


@cv.route("/text-match", strict_slashes=False, methods=["POST"])
class CVHandler(Resource):
    @cv.response(CVTextMatchRequest())
    @cv.expect(doc_upload_parser)
    @cv.response(CVTemplateMatchResponse())
    def post(self):
        """Reads in the file data, performs OCR"""
        args = doc_upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        text = extract(uploaded_file)
        return {
            OCRResponse.TEXT: text,
            OCRResponse.FILE_NAME: uploaded_file.filename,
            OCRResponse.CONTENT_TYPE: uploaded_file.content_type,
            OCRResponse.CONTENT_LENGTH: uploaded_file.content_length,
            OCRResponse.MIMETYPE: uploaded_file.mimetype,
        }
