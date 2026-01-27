import os

# Resolve OpenMP/MKL runtime conflicts on Windows by setting threading
# layer before importing libraries that may load OpenMP (numpy, torch, cv2).
# Prefer MKL using the GNU threading layer to avoid duplicate runtimes.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
# Unsafe fallback to allow duplicate OpenMP libraries if needed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, render_template, request
from esrgan_inference import ESRGANUpscaler

app = Flask(__name__)

model = ESRGANUpscaler("model/RealESRGAN_x4plus.pth")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file selected"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    input_path = "static/uploaded/input.png"
    file.save(input_path)

    output_path, elapsed = model.upscale(input_path)

    return render_template("index.html", input_image=input_path, output_image=output_path, elapsed=elapsed)


if __name__ == "__main__":
    app.run(debug=True)
