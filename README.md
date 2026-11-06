# AI Check

Photo, video ya audio mein AI generated content aur deepfake detect karne ke liye ek local web application.

## Khasiyat (Features)

- **100% Local Inference**: Kisi third-party cloud API ya key ki zaroorat nahi hai. Saare checks aapke computer par locally run hote hain.
- **Multiple Formats**:
  - **Photos**: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`
  - **Videos**: `.mp4`, `.mov`, `.webm`, `.avi`, `.mkv`, `.wmv`
  - **Audios**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
- **Models**:
  - Visual models: `onnx-community/ai-image-detection-ONNX` and `prithivMLmods/Deepfake-Detection-Exp-02-22-ONNX`.
  - Audio: `as1605/Deepfake-audio-detection-V2` (ONNX)

Photo par AI image score 95/100 se strong signal flag hota hai. 65/100 tak aur high deepfake signal ke bina low signal dikhta hai. Beech ke scores aur high deepfake signal review ke liye unclear dikhte hain. Ye starting cutoffs do labeled photos par check kiye gaye hain. Video aur speech ke liye strong signal lagatar do samples mein chahiye. Mixed signals par result unclear dikhta hai. Long video aur audio mein shuru, beech aur end se 10-second windows sample hoti hain. Chhoti files mein poori duration se samples liye jaate hain. Duration unavailable ho toh pehle 30 seconds try hote hain. Black-and-white graphics par low scores review ke liye unclear dikhte hain.

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
