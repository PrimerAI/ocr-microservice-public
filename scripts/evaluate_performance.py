"""
evaluate_performance.py

This script assumes that the OCR service is up and running on port 8000.
This script also assumes it is being invoked from the folder with examples, scripts, and ocr_micro-
service in it.  So `python ./scripts/evaluate_performance.py`.

It will run through a series of examples (from the examples directory) and append a report to
performance_history.txt.
"""
from datetime import datetime
from pathlib import Path
import requests
import sys

example_dir = Path("examples")


def do_ocr(filename) -> str:
    url = "http://localhost:8000/api/v1/ocr"
    files = [
        ('file', open(filename, 'rb'))
    ]
    response = requests.request("POST", url, headers={}, data={}, files=files)
    if not response.ok:
        return None
    return response.json()['text']


def _equals_ignore_case_and_spacing(a: str, b: str) -> bool:
    return "".join(a.lower().split()) == "".join(b.lower().split())


def word_overlap(a:str, b:str) -> float:
    """Compute an off-bleu score defined by the union of tokens. 1.0 is perfect. 0.0 is terrible."""
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    return 1.0 - ((len(a_tokens.difference(b_tokens)) + len(b_tokens.difference(a_tokens))) / float(len(a_tokens.union(b_tokens))))


def simple_doc_evaluation():
    complete_failures = 0  # The number of times the doc actually triggers an exception.
    complete_successes = 0  # The number of times the exact text is extracted (sans spacing)
    average_word_overlap = 0.0

    ground_truth = open(example_dir / "simple_doc.txt", 'rt').read()

    example_filenames = [
        "simple_doc.pdf",
        "simple_doc.png",
        "simple_doc_noisy_scan.png",
        "simple_doc_noisy_scan_bad_crop.png",
        "simple_doc_noisy_scan_bad_crop_rotated.png",
    ]
    for ex_name in example_filenames:
        ex = example_dir / ex_name
        ex_txt = do_ocr(ex)
        if not ex_txt:
            complete_failures += 1
        elif _equals_ignore_case_and_spacing(ex_txt, ground_truth):
            complete_successes += 1
            average_word_overlap += 1.0
        else:
            average_word_overlap += word_overlap(ex_txt, ground_truth)
    average_word_overlap /= len(example_filenames)

    return complete_failures, complete_successes, average_word_overlap


def main(notes="", log=True):
    fail, succeed, overlap = simple_doc_evaluation()
    report = f"{datetime.utcnow()}\t{fail}\t{succeed}\t{overlap}\t{notes}\n"
    if log:
        with open("performance_history.txt", 'a+') as fout:
            fout.write(report)
    print(report)


if __name__=="__main__":
    notes = ""
    if len(sys.argv) > 1:
        notes = sys.argv[1]
    savelog = "--nolog" not in sys.argv
    main(notes, savelog)