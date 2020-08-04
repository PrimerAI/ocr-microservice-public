import numpy
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
from PyPDF2 import PdfFileReader, PdfFileWriter
from pdfminer.high_level import extract_text
import pytesseract
from wand.image import Image as wi
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ocr_microservice.config import config

UPLOAD_FOLDER = config.get_file_upload_path()


def extract(infile: FileStorage):
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
                extracted_texts.append(extract_text_from_image(temp_image_filename))
                os.remove(temp_image_filename)
                os.remove(filepath)

            text = "\n".join(extracted_texts)

    else:
        text = extract_text_from_image(infile, filepath)

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


def _edge_detect(img):
    # A simple wrapper for edge detection that sums the DX/DY components.
    if len(img.shape) > 2 and img.shape[-1] > 1:  # Force grey if it's not already.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # DO NOT LISTEN TO THE DOCS!  Specifying dst != None will return empties.
    edges_x = cv2.Sobel(img.copy(), cv2.CV_32F, 1, 0, ksize=1)
    edges_y = cv2.Sobel(img.copy(), cv2.CV_32F, 0, 1, ksize=1)
    return (numpy.abs(edges_x) + numpy.abs(edges_y)) * 0.5


def match_template_in_image(
        doc_image=None, template_image=None,
        document_filepath: str = None, template_filepath: str = None,
        threshold: float = 0.7,
        max_downsampling: int = 4
):
    """
    Given a document and template (or doc filename and template filename),
    find all the template matches above the threshold in the image across all scales.
    Returns an ARRAY with {"bounding_box":[x,y,w,h], "confidence":##.##}.
    """
    assert (doc_image is not None) or (document_filepath is not None)
    assert (template_image is not None) or (template_filepath is not None)
    if document_filepath:
        doc_image = cv2.imread(document_filepath)
    if template_filepath:
        template_image = cv2.imread(template_filepath)

    # Make greyscale explicitly, though many of our docs will be grey anyway.
    doc_image = cv2.cvtColor(doc_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # Our matching method will be basically a dot product, so we want edge responses.
    template_edges = _edge_detect(template_image)

    # For a multitude of scales, retry the finds of the template_image on the doc_image.
    # Since scaling up the template doesn't get us additional resolution and costs us more compute,
    # we instead scale down the doc image by integers.  Since this is naive matching, we don't have
    # to worry about optimizing for matching harmonics or anything like that and can constrain our
    # resizing to integers!
    matches = list()
    scale = 1
    # When our doc_image is less than the template size on any axis, we know it can't match.
    while doc_image.shape[0]//scale > template_image.shape[0] \
            and doc_image.shape[1]//scale > template_image.shape[1] \
            and scale < max_downsampling:
        # Yes, PIL is easier to use for resizing, but then we have to convert back and forth.
        search_image = numpy.zeros((doc_image.shape[0]//scale, doc_image.shape[1]//scale, 1), dtype=numpy.uint8)
        cv2.resize(doc_image, dsize=None, dst=search_image, fx=1.0/float(scale), fy=1.0/float(scale))

        # Compute the edges _NOW_ rather than before reduction so we actually get fovation.
        search_edges = _edge_detect(search_image)

        # Template match, resize our response, and accumulate it.
        scaled_match_response = cv2.matchTemplate(search_edges, template_edges, cv2.TM_CCORR_NORMED)

        # DEBUG: Show template:
        #Image.fromarray(numpy.clip(255 * scaled_match_response[:, :], 0, 255).astype(numpy.uint8)).show()

        # Go over the scaled match response and add the bounding boxes if they're above threshold.
        # Don't forget to resize the bounding boxes by the inverse of the scale factor.
        # NOTE: This gives a _single_ response per resolution.  If we eventually care about more,
        # we will have to do things differently, but for detecting if text exists, this should be
        # just fine.
        max_response_position = numpy.unravel_index(
            scaled_match_response.argmax(),
            scaled_match_response.shape
        )
        max_response = scaled_match_response[max_response_position]
        if max_response > threshold:
            matches.append({
                "bounding_box": [
                    # NOTE THE COORDINATE CHANGE!
                    # OpenCV uses column-major because we can't have nice things.
                    max_response_position[1]*scale,  # x
                    max_response_position[0]*scale,  # y
                    template_edges.shape[1]*scale,  # width
                    template_edges.shape[0]*scale,  # height
                ],
                "confidence": max_response,
                "scale": scale,
            })

        # Don't forget to update the scale.
        scale += 1

    return matches


def match_text_in_image(
        text: str,
        document_filename: str = None,
        doc_image=None,
        threshold: float = 0.0,
        text_size: int = 12,
        font: str = "Arial.ttf"
):
    # Assuming 96 DPI, 6pt = 8px.  12pt = 16px.
    # Allocate a drawing area fitting of the text size.
    bg_color = (0, 0, 0)
    fg_color = (255, 255, 255)
    template_size = (int(len(text)*text_size), int((1+text.count('\n'))*text_size*1.5))

    template = Image.new("RGB", template_size, bg_color)
    fnt = ImageFont.truetype(font, text_size)
    ctx = ImageDraw.Draw(template)

    # Draw to the template context
    ctx.multiline_text((0, 0), text, font=fnt, fill=fg_color)

    # Convert the template to a cv2 image (which is basically a numpy array).
    template = numpy.asarray(template, dtype=numpy.uint8)

    return match_template_in_image(
        document_filepath=document_filename,
        doc_image=doc_image,
        template_image=template,
        threshold=threshold
    )

def _debug_highlight_regions(doc_filename, matches):
    img = Image.open(doc_filename)
    ctx = ImageDraw.Draw(img)
    for m in matches:
        x, y, w, h = m['bounding_box']
        ctx.rectangle([x, y, x+w, y+h], fill=None, outline="red")
    print(matches)
    img.show()