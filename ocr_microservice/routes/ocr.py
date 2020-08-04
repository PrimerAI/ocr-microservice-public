"""Endpoint for checking service health."""
from flask_restplus import Resource
from ocr_microservice.schemas import OCRResponse
from primer_micro_utils.namespace import Namespace
from werkzeug.datastructures import FileStorage

from ocr_microservice.common import extract

ocr = Namespace("ocr", description="Ping Namespace and Endpoints")

upload_parser = ocr.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@ocr.route("/", strict_slashes=False, methods=["POST"])
class OCRHandler(Resource):
    @ocr.response(OCRResponse())
    @ocr.expect(upload_parser)
    def post(self):
        """Reads in the file data, performs OCR"""
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        text = extract(uploaded_file)
        return {
            OCRResponse.TEXT: text,
            OCRResponse.FILE_NAME: uploaded_file.filename,
            OCRResponse.CONTENT_TYPE: uploaded_file.content_type,
            OCRResponse.CONTENT_LENGTH: uploaded_file.content_length,
            OCRResponse.MIMETYPE: uploaded_file.mimetype,
        }


