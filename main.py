import os
from dotenv import load_dotenv
from flask import Flask, make_response, send_from_directory
from flask_cors import CORS

from api.routes import api_bp
from views.template import KAPARSH_FRONTEND

load_dotenv()

app = Flask(__name__)
CORS(app)

app.register_blueprint(api_bp, url_prefix="/api")

@app.route("/manifest.json", methods=["GET"])
def serve_manifest():
    return send_from_directory(app.static_folder, "manifest.json")

@app.route("/sw.js", methods=["GET"])
def serve_sw():
    response = make_response(send_from_directory(app.static_folder, "sw.js"))
    response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route("/", methods=["GET"])
def serve_frontend():
    response = make_response(KAPARSH_FRONTEND)
    response.headers["Content-Type"] = "text/html"
    return response

if __name__ == "__main__":
    app.run(debug=True)