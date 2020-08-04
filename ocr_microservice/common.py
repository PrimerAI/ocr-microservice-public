import os
import cv2
from PIL import Image
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdfminer.high_level import extract_text
import pytesseract
from wand.image import Image as wi
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ocr_microservice.config import config

UPLOAD_FOLDER = config.get_file_upload_path()


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
        if len(text) < 1:
            # no embedded text; convert to image, splitting into pages to avoid
            # memory limits for high resolution conversions

            pdf = PdfFileReader(infile)

            page_filepaths = list()
            extracted_texts = list()

            for page in range(pdf.getNumPages()):
                pdf_writer = PdfFileWriter()
                pdf_writer.addPage(pdf.getPage(page))

                page_filepath = f"page_{page + 1}"
                with open(page_filepath, "wb") as f:
                    pdf_writer.write(f)
                    page_filepaths.append(page_filepath)

            for filepath in page_filepaths:
                temp_image_filename = "page.png"  # TODO:  use tempfile
                with wi(filename=filepath, resolution=900).convert(
                        "png"
                ) as pdf_image:
                    wi(image=pdf_image).save(filename=temp_image_filename)
                extracted_texts.append(self.extract_text_from_image(temp_image_filename))
                os.remove(temp_image_filename)
                os.remove(filepath)

            text = "\n".join(extracted_texts)

    else:
        text = self.extract_text_from_image(infile, filepath)

    os.remove(filepath)
    return text


def extract_text_from_image(infile, filepath):
    """Process an image and return any text extracted"""

    # load the example image and convert it to grayscale
    image = cv2.imread(filepath)

    # handle tifs since they dont display in web post ocr process
    if infile.filename.endswith(".tif"):
        im = Image.open(filepath)
        filenamefix = filepath.rsplit(".", 1)[0] + ".jpg"
        filenamefix = filenamefix.rsplit("/", 1)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filenamefix)
        out = im.convert("RGB")
        out.save(filepath, "JPEG", quality=80)

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a small gaussian blur before Otsu's threshold
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

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

    return text