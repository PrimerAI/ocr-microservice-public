import numpy
from PIL import Image, ImageDraw, ImageFont
from unittest import TestCase

from ocr_microservice.common import match_template_in_image


class Test(TestCase):
    def test_match_template_in_image(self):
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Draw a box and a circle.  Find them.
        doc = Image.new("RGB", (600, 800), white)
        rect_template = Image.new("RGB", (100, 100), white)
        circle_template = Image.new("RGB", (100, 100), white)

        # Draw all our shapes to their respective templates and to the document.
        # Exclude the diagonal line because want a negative case.
        ctx = ImageDraw.Draw(rect_template)
        ctx.rectangle([5, 5, 25, 25], outline=black)  # x1 y1 x2 y2

        ctx = ImageDraw.Draw(circle_template)
        ctx.chord([5, 5, 15, 15], 0, 360, fill=None, outline=black)

        ctx = ImageDraw.Draw(doc)
        ctx.rectangle([10, 10, 30, 30], fill=None, outline=black)
        ctx.chord([50, 50, 60, 60], 0, 360, fill=None, outline=black)

        # Convert everything to a cv2 image.
        doc = numpy.asarray(doc, dtype=numpy.uint8)
        rect_template = numpy.asarray(rect_template, dtype=numpy.uint8)
        circle_template = numpy.asarray(circle_template, dtype=numpy.uint8)

        rect_findings = match_template_in_image(doc_image=doc, template_image=rect_template)
        self.assertTrue(len(rect_findings) > 0, "Rect template was not found on the doc!")

        circle_findings = match_template_in_image(doc_image=doc, template_image=circle_template)
        self.assertTrue(len(circle_findings) > 0, "Circle template was not found on the doc!")
