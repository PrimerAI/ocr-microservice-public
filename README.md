# OCR Microservice

Upload a PDF or image for OCR

## Local Dev Setup

```
make setup-local
workon ocr-microservice
```

## Build the App

```
make docker-build

```

## Running the App

```
make docker-start

```

Then open up browser to http://localhost:8000/api_doc/v1/ to see the swagger docs

## Using/Testing

### Postman

Create a Request with the following settings:
POST http://localhost:8000/api/v1/ocr/
Body: form-data
KEY: file
hover over the key name to see the drop-down, and select File
VALUE: browse to your desired file

### cURL

```
curl --location --request POST 'http://localhost:8000/api/v1/ocr/' \
--form 'file=@/path/to/file.pdf'
```

### python

```
import requests

url = "http://localhost:8000/api/v1/ocr"

payload = {}
files = [
  ('file', open('/Users/johnsullivan/Documents/Army/G-2/transfer_15073_files_ab3bded1/UNCLASSIFIED DATA_pdf/24th MI ACE RUS Artillery troops fire Tyulpan heavy mortars in southern Russia drills U 20200331.pdf','rb'))
]
headers= {}

response = requests.request("POST", url, headers=headers, data = payload, files = files)

print(response.text.encode('utf8'))

```

### Sample Test Data

- examples/ara-1.jpg
- examples/ara-2.jpg
- examples/eng-1.png
- examples/eng-2.jpg

## Built With

```bash
Python
Flask
Pytesseract
OpenCV
Bootstrap
Docker
```

## Resources

Here are some helpful resources I used for this project.

- [Deep Learning based Text Recognition (OCR) using Tesseract and OpenCV](https://www.learnopencv.com/deep-learning-based-text-recognition-ocr-using-tesseract-and-opencv/)
- [Using Tesseract OCR with Python](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)
