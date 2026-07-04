import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = FastAPI(title="Image Upscaling API")

# Initialize the model at startup
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
try:
    upsampler = RealESRGANer(
        scale=4,
        model_path='model/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
except Exception as e:
    print(f"Warning: Failed to load RealESRGAN model: {e}")
    upsampler = None

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    if not upsampler:
        raise HTTPException(status_code=500, detail="Upscaler model not loaded.")
        
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    # Upscale
    output, _ = upsampler.enhance(img, outscale=4)
    
    # Encode and return
    _, encoded_img = cv2.imencode('.png', output)
    return Response(content=encoded_img.tobytes(), media_type="image/png")
