import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Image Analytics Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main { background: #f7f9fc; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #111827, #26364d);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0 0 .35rem 0; font-size: 2.2rem; }
    .hero p { margin: 0; color: #dbe5f0; font-size: 1rem; }
    .card {
        background: white; border: 1px solid #e5eaf0; border-radius: 15px;
        padding: 1.1rem 1.25rem; margin: .5rem 0 1rem 0;
        box-shadow: 0 3px 14px rgba(15,23,42,.04);
    }
    .tag {
        display:inline-block; padding:.22rem .55rem; border-radius:999px;
        background:#eef2ff; color:#3730a3; font-size:.78rem; font-weight:700;
    }
    .small { color:#64748b; font-size:.9rem; }
    div[data-testid="stMetric"] {
        background:white; border:1px solid #e5eaf0; border-radius:12px; padding:.7rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def read_image(uploaded):
    if uploaded is None:
        return None
    try:
        data = uploaded.getvalue()
        if not data:
            raise ValueError("The uploaded file is empty.")
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(pil)
    except Exception as e:
        st.error(f"Could not read the image: {e}")
        return None

def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_temp_image(img, suffix=".jpg"):
    import tempfile
    fd, path = tempfile.mkstemp(suffix=suffix)
    Path(path).write_bytes(cv2.imencode(".jpg", rgb_to_bgr(img))[1].tobytes())
    return path

def explanation(title, text):
    st.markdown(f'<div class="card"><span class="tag">{title}</span><p class="small" style="margin-top:.65rem">{text}</p></div>', unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🔬 Image Analytics Lab</h1>
  <p>Interactive demonstrations of classical and deep-learning image & face analytics.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Navigation")
    mode = st.radio(
        "Choose a technique",
        ["🏠 Overview", "🎯 Template Matching", "👤 Viola–Jones", "🧠 DeepFace", "🔐 FaceNet Verification"],
    )
    st.divider()
    st.caption("College Project / Demo")
    st.caption("Python • OpenCV • Streamlit • Deep Learning")

# ---------- Overview ----------
if mode == "🏠 Overview":
    st.subheader("Welcome")
    st.write("Upload images and run real analytics directly in the browser. Each module includes an explanation, controls, visual output, and error handling.")

    cols = st.columns(4)
    cards = [
        ("🎯 Template Matching", "Find a known image patch inside a larger image using OpenCV correlation methods."),
        ("👤 Viola–Jones", "Detect faces with a Haar-cascade classifier using the classic Viola–Jones framework."),
        ("🧠 DeepFace", "Analyze face attributes such as age, gender, emotion and race using pretrained models."),
        ("🔐 FaceNet Verification", "Compare two face images using FaceNet embeddings and a distance threshold."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h4>{title}</h4><p class="small">{desc}</p></div>', unsafe_allow_html=True)

    st.info("Tip: For DeepFace and FaceNet, the first run can take longer because model weights may need to be downloaded/cached.")

# ---------- Template Matching ----------
elif mode == "🎯 Template Matching":
    st.subheader("🎯 Template Matching")
    explanation("Algorithm", "OpenCV slides a template over the source image and computes a similarity score at each location. The highest-scoring region is selected as the best match.")

    c1, c2 = st.columns(2)
    with c1:
        source_file = st.file_uploader("Upload source image", type=["jpg","jpeg","png"], key="tm_source")
    with c2:
        template_file = st.file_uploader("Upload template image", type=["jpg","jpeg","png"], key="tm_template")

    source = read_image(source_file)
    template = read_image(template_file)

    if source is not None and template is not None:
        if template.shape[0] > source.shape[0] or template.shape[1] > source.shape[1]:
            st.error("Template must be smaller than (or equal to) the source image.")
        else:
            method_name = st.selectbox(
                "Matching method",
                ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"],
            )
            threshold = st.slider("Match confidence threshold", 0.0, 1.0, 0.70, 0.01)

            if st.button("Run Template Matching", type="primary"):
                try:
                    methods = {
                        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
                        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
                        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
                    }
                    src_gray = cv2.cvtColor(rgb_to_bgr(source), cv2.COLOR_BGR2GRAY)
                    tpl_gray = cv2.cvtColor(rgb_to_bgr(template), cv2.COLOR_BGR2GRAY)
                    result = cv2.matchTemplate(src_gray, tpl_gray, methods[method_name])
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                    if method_name == "TM_SQDIFF_NORMED":
                        score = 1.0 - min_val
                        top_left = min_loc
                    else:
                        score = max_val
                        top_left = max_loc

                    h, w = tpl_gray.shape
                    output = source.copy()
                    if score >= threshold:
                        cv2.rectangle(output, top_left, (top_left[0] + w, top_left[1] + h), (255, 0, 0), 3)

                    a, b, c = st.columns(3)
                    a.metric("Best score", f"{score:.3f}")
                    b.metric("Threshold", f"{threshold:.2f}")
                    c.metric("Status", "Match found" if score >= threshold else "Below threshold")

                    r1, r2 = st.columns(2)
                    with r1:
                        st.image(source, caption="Source", use_container_width=True)
                    with r2:
                        st.image(output, caption="Detected template region", use_container_width=True)

                    if score < threshold:
                        st.warning("The best location did not meet the selected confidence threshold.")
                except Exception as e:
                    st.error(f"Template matching failed: {e}")

# ---------- Viola Jones ----------
elif mode == "👤 Viola–Jones":
    st.subheader("👤 Viola–Jones Face Detection")
    explanation("Algorithm", "Viola–Jones uses Haar-like features, an integral image for fast feature computation, AdaBoost-selected weak classifiers, and a cascade structure to rapidly reject non-face regions.")

    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="vj")
    img = read_image(uploaded)

    if img is not None:
        scale = st.slider("Scale factor", 1.05, 1.50, 1.10, 0.01)
        neighbors = st.slider("Minimum neighbors", 1, 10, 5)
        if st.button("Detect Faces", type="primary"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                classifier = cv2.CascadeClassifier(cascade_path)
                if classifier.empty():
                    raise RuntimeError("OpenCV Haar cascade could not be loaded.")

                bgr = rgb_to_bgr(img)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbors, minSize=(30,30))

                output = img.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(output, (x,y), (x+w,y+h), (0,255,0), 3)

                st.metric("Faces detected", len(faces))
                st.image(output, caption="Viola–Jones result", use_container_width=True)
            except Exception as e:
                st.error(f"Face detection failed: {e}")

# ---------- DeepFace ----------
elif mode == "🧠 DeepFace":
    st.subheader("🧠 DeepFace Analysis")
    explanation("Algorithm", "DeepFace provides pretrained deep-learning models for facial attribute analysis. The app uses the DeepFace Analyze API to estimate attributes from detected faces.")

    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"], key="deepface")
    img = read_image(uploaded)

    if img is not None:
        actions = st.multiselect(
            "Attributes to analyze",
            ["age", "gender", "emotion"],
            default=["emotion"],
        )
        detector = st.selectbox("Face detector backend", ["opencv", "retinaface", "mtcnn", "mediapipe"])

        if st.button("Run DeepFace Analysis", type="primary"):
            if not actions:
                st.warning("Select at least one attribute.")
            else:
                try:
                    from deepface import DeepFace
                    path = save_temp_image(img)
                    with st.spinner("Running DeepFace models..."):
                        result = DeepFace.analyze(
                            img_path=path,
                            actions=actions,
                            detector_backend=detector,
                            enforce_detection=True,
                            silent=True,
                        )

                    results = result if isinstance(result, list) else [result]
                    st.success(f"Analysis completed for {len(results)} detected face(s).")

                    for i, face in enumerate(results, start=1):
                        st.markdown(f"#### Face {i}")
                        cols = st.columns(len(actions))
                        for col, action in zip(cols, actions):
                            value = face.get(action, "N/A")
                            if isinstance(value, dict):
                                value = max(value.items(), key=lambda x: x[1])[0]
                            with col:
                                st.metric(action.title(), str(value))

                        region = face.get("region", {})
                        if region:
                            st.caption(f"Face region: x={region.get('x')}, y={region.get('y')}, w={region.get('w')}, h={region.get('h')}")
                except ImportError:
                    st.error("DeepFace is not installed. Run: pip install -r requirements.txt")
                except Exception as e:
                    st.error(f"DeepFace analysis failed: {e}")
                    st.info("Make sure the image contains a clear face and that the selected detector/model dependencies are available.")

# ---------- FaceNet ----------
elif mode == "🔐 FaceNet Verification":
    st.subheader("🔐 FaceNet Face Verification")
    explanation("Algorithm", "FaceNet maps a face image to a compact embedding vector. Verification compares two embeddings using a distance metric; smaller distance means greater similarity.")

    c1, c2 = st.columns(2)
    with c1:
        file1 = st.file_uploader("Upload Image 1", type=["jpg","jpeg","png"], key="fn1")
    with c2:
        file2 = st.file_uploader("Upload Image 2", type=["jpg","jpeg","png"], key="fn2")

    img1 = read_image(file1)
    img2 = read_image(file2)

    if img1 is not None and img2 is not None:
        threshold = st.slider("Euclidean distance threshold", 0.2, 2.0, 1.10, 0.05)

        if st.button("Verify Faces", type="primary"):
            try:
                from facenet_pytorch import MTCNN, InceptionResnetV1
                import torch
                from PIL import Image as PILImage

                device = "cuda" if torch.cuda.is_available() else "cpu"
                with st.spinner("Loading FaceNet model and computing embeddings..."):
                    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
                    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

                    face1 = mtcnn(PILImage.fromarray(img1))
                    face2 = mtcnn(PILImage.fromarray(img2))

                    if face1 is None or face2 is None:
                        raise ValueError("A face could not be detected in one or both images.")

                    with torch.no_grad():
                        emb1 = model(face1.unsqueeze(0).to(device))
                        emb2 = model(face2.unsqueeze(0).to(device))
                        distance = torch.norm(emb1 - emb2, p=2).item()

                verified = distance <= threshold
                a, b, c = st.columns(3)
                a.metric("Embedding distance", f"{distance:.4f}")
                b.metric("Threshold", f"{threshold:.2f}")
                c.metric("Result", "Same person" if verified else "Different / uncertain")

                r1, r2 = st.columns(2)
                with r1:
                    st.image(img1, caption="Image 1", use_container_width=True)
                with r2:
                    st.image(img2, caption="Image 2", use_container_width=True)

                if verified:
                    st.success("The two embeddings are within the selected threshold.")
                else:
                    st.warning("The embeddings are outside the selected threshold. Threshold choice affects verification.")
            except ImportError:
                st.error("FaceNet dependencies are not installed. Run: pip install -r requirements.txt")
            except Exception as e:
                st.error(f"Face verification failed: {e}")
                st.info("Use clear, front-facing face images. The first model run may also download pretrained weights.")

st.divider()
st.caption("Image Analytics Lab • Educational demonstration • Do not use biometric outputs as a sole basis for real-world decisions.")
