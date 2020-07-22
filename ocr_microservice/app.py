from flask_restplus import Api

from ocr_microservice.routes import ocr
from ocr_microservice.routes import ping

from primer_micro_utils.app import generate_app

__source__ = ""

UPLOAD_FOLDER = "./static/uploads"


def create_app():
    """Create the flask app"""
    # app = Flask(__name__)
    # app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    # app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    # Initialize the RestPLUS API.
    v1_api = Api(prefix='/api/v1', doc='/api_doc/v1/')

    v1_api.add_namespace(ocr)
    v1_api.add_namespace(ping)

    app = generate_app(v1_api, name=__name__, identity_func=anon_identity_func)
    return app


def anon_identity_func():
    return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
