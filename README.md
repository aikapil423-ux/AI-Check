# AI Check

Photo, video ya audio mein AI generated content aur deepfake detect karne ke liye ek local web application.

## Khasiyat (Features)

- **100% Local Inference**: Kisi third-party cloud API ya key ki zaroorat nahi hai. Saare checks aapke computer par locally run hote hain.
- **Multiple Formats**:
  - **Photos**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`
  - **Videos**: `.mp4`, `.mov`, `.webm`, `.avi`, `.mkv`, `.wmv`
  - **Audios**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
- **Models**:
  - Visual: `onnx-community/ai-image-detect-distilled-ONNX` (quantized ONNX)
  - Audio: `as1605/Deepfake-audio-detection-V2` (ONNX)

## Setup & Kaise Chalayein

### Windows (Quick Start)
1. [`Start AI Check.cmd`](./Start%20AI%20Check.cmd) par double-click karein.
2. Browser mein [http://127.0.0.1:4173](http://127.0.0.1:4173) open karein.

### Terminal Se

```bash
npm install
npm start
```

Browser mein `http://127.0.0.1:4173` kholein.
