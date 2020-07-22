"""Setup the ocr_microservice Service."""

from pip._internal.cli.parser import ConfigOptionParser
from pip._internal.req import parse_requirements
from setuptools import setup, find_packages


def _parse(filename):
    return [str(r.requirement) for r in parse_requirements(filename, session=False)]


pip_conf_parser = ConfigOptionParser(name="local")
pip_conf_parser.config.load()
# Get the `extra_index_url`, prioritizes the env var `PIP_EXTRA_INDEX_URL`
pip_config_list = pip_conf_parser._get_ordered_configuration_items()
extra_index_url = dict(pip_config_list).get("extra-index-url")

setup(
    name="ocr_microservice",
    version=open("VERSION").read().strip(),
    author="Primer AI",
    author_email="engineering@primer.ai",
    description="ocr-microservice",
    url="https://github.com/PrimerAI/ocr-microservice",
    packages=find_packages(exclude=("tests")),
    install_requires=_parse("requirements.txt"),
    dependency_links=[extra_index_url + "primer-micro-utils",],
)
