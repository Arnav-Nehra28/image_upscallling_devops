from esrgan_inference import ESRGANUpscaler

if __name__ == '__main__':
    model_path = 'model/RealESRGAN_x4plus.pth'
    up = ESRGANUpscaler(model_path, verbose=True)
    input_path = 'static/uploaded/input.png'
    out, elapsed = up.upscale(input_path, save_path='static/uploaded/test_output.png')
    print(f'Done: {out} ({elapsed:.2f}s)')
