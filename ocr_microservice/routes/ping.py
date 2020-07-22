"""Endpoint for checking service health."""

from flask_restplus import Resource

from ocr_microservice.schemas import PingResponse

from primer_micro_utils.namespace import Namespace

ping = Namespace("ping", description="Ping Namespace and Endpoints")


@ping.route("/", strict_slashes=False)
class PingHandler(Resource):
    @ping.response(PingResponse())
    def get(self):
        """Returns a status as green if this service is up and running."""
        return {"status": "green"}
