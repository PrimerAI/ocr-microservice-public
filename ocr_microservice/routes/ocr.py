"""Endpoint for checking service health."""
import os

import cv2

from PIL import Image

from PyPDF2 import PdfFileReader, PdfFileWriter

from flask import Flask, render_template, request

from pdfminer.high_level import extract_text

import pytesseract

from wand.image import Image as wi

from werkzeug.utils import secure_filename

from flask_restplus import Resource

from ocr_microservice.schemas import OCRResponse

from primer_micro_utils.namespace import Namespace

from werkzeug.datastructures import FileStorage

UPLOAD_FOLDER = "/code/"

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
        text = self.extract(uploaded_file)
        return {
            OCRResponse.TEXT: text,
            OCRResponse.FILE_NAME: uploaded_file.filename,
            OCRResponse.CONTENT_TYPE: uploaded_file.content_type,
            OCRResponse.CONTENT_LENGTH: uploaded_file.content_length,
            OCRResponse.MIMETYPE: uploaded_file.mimetype,
        }

    def extract(self, infile: FileStorage):
        """Process the uploaded file, and return any extracted text"""

        # create a secure filename
        filename = secure_filename(infile.filename)

        # save file to /static/uploads
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        infile.save(filepath)

        if filepath.endswith(".pdf"):
            # try to extract text from the PDF directly
            text = extract_text(infile)

            # TODO:  Better heuristic for failure
            if len(text) < 50:
                # no embedded text; convert to image, splitting into pages to avoid
                # memory limits for high resolution conversions

                pdf = PdfFileReader(infile, strict=False)

                page_filepaths = list()
                extracted_texts = list()

                for page in range(pdf.getNumPages()):
                    pdf_writer = PdfFileWriter()
                    pdf_writer.addPage(pdf.getPage(page))

                    page_filepath = f"page_{page+1}"
                    with open(page_filepath, "wb") as f:
                        pdf_writer.write(f)
                        page_filepaths.append(page_filepath)

                for fp in page_filepaths:
                    temp_jpg_filepath = "page.jpg"  # TODO:  use tempfile
                    with wi(filename=fp, resolution=900).convert(
                            "jpeg"
                    ) as pdf_image:
                        wi(image=pdf_image).save(filename=temp_jpg_filepath)
                    extracted_texts.append(self.extract_text_from_image(infile, temp_jpg_filepath))
                    os.remove(temp_jpg_filepath)
                    os.remove(fp)

                text = "\n".join(extracted_texts)

        else:
            text = self.extract_text_from_image(infile, filepath)

        os.remove(filepath)
        return text

    def extract_text_from_image(self, infile, filepath):
        """Process an image and return any text extracted"""

        # load the example image and convert it to grayscale
        image = cv2.imread(filepath)
        # handle tifs since they dont display in web post ocr process
        filenamefix = ""

        if infile.filename.endswith(".tif"):
            im = Image.open(filepath)
            filenamefix = filepath.rsplit(".", 1)[0] + ".jpg"
            filenamefix = filenamefix.rsplit("/", 1)[1]
            filepathfix = os.path.join(UPLOAD_FOLDER, filenamefix)
            out = im.convert("RGB")
            out.save(filepathfix, "JPEG", quality=80)
        # convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply thresholding to preprocess the image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # apply median blurring to remove any blurring
        gray = cv2.medianBlur(gray, 3)

        # save the processed image in the /static/uploads directory
        ofilename = os.path.join(UPLOAD_FOLDER, "{}.png".format(os.getpid()))
        cv2.imwrite(ofilename, gray)

        # perform OCR on the processed image
        text = pytesseract.image_to_string(Image.open(ofilename), lang="eng")

        # remove the processed image
        os.remove(ofilename)
        if filenamefix != "":
            filename = filenamefix

        return text
