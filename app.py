from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sys
from PIL import Image
from pdfminer.high_level import extract_text
from PyPDF2 import PdfFileReader, PdfFileWriter
import pytesseract
import argparse
import cv2
from wand.image import Image as wi

__source__ = ''

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        # create a secure filename
        filename = secure_filename(f.filename)

        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        if filepath.endswith('.pdf'):
            # try to extract text from the PDF directly
            text = extract_text(filepath)

            # TODO:  Better heuristic for failure
            if len(text) < 50:
                # no embedded text; convert to image, splitting into pages to avoid
                # memory limits for high resolution conversions

                pdf = PdfFileReader(filepath)

                page_filepaths = list()
                extracted_texts = list()

                for page in range(pdf.getNumPages()):
                    pdf_writer = PdfFileWriter()
                    pdf_writer.addPage(pdf.getPage(page))

                    page_filepath = f'page_{page+1}'
                    with open(page_filepath, 'wb') as f:
                        pdf_writer.write(f)
                        page_filepaths.append(page_filepath)

                for filepath in page_filepaths:
                    temp_jpg_filepath = "page.jpg"  # TODO:  use tempfile
                    with wi(filename=filepath, resolution=900).convert("jpeg") as pdf_image:
                        wi(image=pdf_image).save(filename=temp_jpg_filepath)
                    extracted_texts.append(extract_text_from_image(temp_jpg_filepath))
                    os.remove(temp_jpg_filepath)

                text = "\n".join(extracted_texts)

        else:
            text = extract_text_from_image(filepath)
        return render_template("uploaded.html", displaytext=text, fname=filename)


# TODO -- Put this in a module, not app.py
def extract_text_from_image(filepath):

    # load the example image and convert it to grayscale
    image = cv2.imread(filepath)
    # handle tifs since they dont display in web post ocr process
    filenamefix = ''

    if filepath.endswith('.tif'):
        im = Image.open(filepath)
        filenamefix = filepath.rsplit('.',1)[0]+'.jpg'
        filenamefix = filenamefix.rsplit('/',1)[1]
        filepathfix = os.path.join(app.config['UPLOAD_FOLDER'], filenamefix)
        out = im.convert("RGB")
        out.save(filepathfix, "JPEG", quality=80)
    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding to preprocess the image
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply median blurring to remove any blurring
    gray = cv2.medianBlur(gray, 3)

    # save the processed image in the /static/uploads directory
    ofilename = os.path.join(
        app.config['UPLOAD_FOLDER'], "{}.png".format(os.getpid()))
    cv2.imwrite(ofilename, gray)

    # perform OCR on the processed image
    text = pytesseract.image_to_string(
        Image.open(ofilename), lang="eng")

    # remove the processed image
    os.remove(ofilename)
    if filenamefix != '':
        filename = filenamefix

    return text

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
