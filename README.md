# Image Analytics Lab

A modern Streamlit college-project demo for Image & Video Analytics.

## Features

1. Template Matching — OpenCV `matchTemplate`
2. Viola–Jones Face Detection — OpenCV Haar cascade
3. DeepFace Analysis — pretrained deep-learning face analysis
4. FaceNet Face Verification — FaceNet embeddings + Euclidean distance

## Run locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Start:
```bash
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Deployment

This project includes `packages.txt` and `runtime.txt` for common Streamlit Community Cloud deployments.

1. Push this folder to GitHub.
2. Create a Streamlit Community Cloud app.
3. Select `app.py` as the entry point.
4. Deploy.

DeepFace and FaceNet are substantially heavier than the OpenCV modules. On a small cloud instance, the first model load can be slow and memory limits may matter.

## Notes

- DeepFace downloads/caches model weights on first use.
- FaceNet uses `facenet-pytorch` with the pretrained VGGFace2 InceptionResnetV1 model.
- FaceNet verification threshold is configurable in the UI. It is an educational demo threshold, not a universal biometric decision threshold.
- The application does not permanently store uploaded images.
