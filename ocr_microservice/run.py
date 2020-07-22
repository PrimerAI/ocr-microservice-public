"""Executes the long-running web app"""

from primer_micro_utils.monkey import patch  # noqa

patch(enable_gevent=True, flask=True, logging=True, psycopg=True, requests=True)  # noqa

# pylint: disable=wrong-import-order,wrong-import-position

import logging  # noqa pylint: disable=C0413,C0411

from ocr_microservice.app import create_app  # noqa pylint: disable=C0413,C0411

app = create_app()

if __name__ == "__main__":
    # For local development only
    app.run(port=8000, debug=True)
else:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)  # pylint: disable=no-member
