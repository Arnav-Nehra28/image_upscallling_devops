import os
import requests
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

ML_URL = os.environ.get("ML_URL", "http://127.0.0.1:8001/upscale")

UPLOAD_DIR = "static/uploads"
OUTPUT_DIR = "static/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # 🔴 FIX: accept "file" (matches your HTML)
    if "file" not in request.files:
        return "No image uploaded", 400

    image = request.files["file"]

    if image.filename == "":
        return "Empty filename", 400

    input_path = os.path.join(UPLOAD_DIR, image.filename)
    output_path = os.path.join(OUTPUT_DIR, image.filename)

    image.save(input_path)

    # Send to ML (running locally on your machine)
    with open(input_path, "rb") as f:
        r = requests.post(
            ML_URL,
            files={"file": f},
            timeout=300
        )

    if r.status_code != 200:
        return f"ML error: {r.text}", 500

    with open(output_path, "wb") as f:
        f.write(r.content)

    return render_template(
        "index.html",
        input_image=f"/{input_path}",
        output_image=f"/{output_path}",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
