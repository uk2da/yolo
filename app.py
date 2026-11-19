from collections import Counter
from io import BytesIO

import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="YOLO Vision | AI 이미지 분석",
    page_icon="👁️",
    layout="wide",
)

try:
    from ultralytics import YOLO
except ModuleNotFoundError as error:
    if error.name != "ultralytics":
        raise
    st.error("YOLO 추론을 사용하려면 `ultralytics` 패키지가 필요합니다.")
    st.code("python -m pip install ultralytics", language="bash")
    st.stop()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17231f;
        --muted: #718078;
        --green: #2f7657;
        --mint: #e8f3ec;
        --line: #e8ede9;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Noto Sans KR', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, #edf6ef 0, transparent 34%),
            #f8faf8;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #f1f6f2;
        border-right: 1px solid var(--line);
    }
    .eyebrow {
        color: var(--green);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .65rem;
    }
    .hero-title {
        font-size: clamp(2rem, 5vw, 3.35rem);
        font-weight: 800;
        letter-spacing: -.055em;
        line-height: 1.14;
        margin: 0;
        color: var(--ink);
    }
    .hero-copy {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.7;
        margin-top: .9rem;
        max-width: 650px;
    }
    .section-label {
        color: var(--green);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .1em;
        margin-bottom: .35rem;
    }
    .section-title {
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -.025em;
        margin: 0 0 .35rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        box-shadow: 0 5px 18px rgba(34, 66, 48, .035);
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }
    div[data-testid="stFileUploader"], div[data-testid="stCameraInput"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .5rem;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 3rem;
        }
        .hero-copy {
            font-size: .95rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


with st.sidebar:
    st.markdown("## 👁️ YOLO Vision")
    st.caption("이미지 속 사물을 빠르게 찾아드립니다.")
    st.divider()
    st.markdown("### 분석 설정")
    model_path = st.text_input("모델 경로", value="yolo26n.pt")
    confidence = st.slider(
        "최소 신뢰도",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="값을 높이면 더 확실한 탐지만 표시합니다.",
    )
    image_size = st.select_slider(
        "이미지 분석 크기",
        options=[320, 480, 640, 800, 1024],
        value=640,
        help="큰 이미지일수록 세밀하지만 분석 시간이 더 걸릴 수 있습니다.",
    )
    st.divider()
    st.caption("처음 실행하면 모델 파일을 자동으로 준비합니다.")


st.markdown('<div class="eyebrow">AI object detection</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title">사진 한 장으로 시작하는<br>스마트한 이미지 분석</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-copy">휴대폰으로 바로 촬영하거나 갤러리에서 사진을 선택하세요. '
    'YOLO가 이미지 속 사물을 찾아 위치와 신뢰도를 보여드립니다.</p>',
    unsafe_allow_html=True,
)
st.write("")

st.markdown('<div class="section-label">01 / IMAGE</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">분석할 사진을 선택해 주세요</div>', unsafe_allow_html=True)
input_method = st.radio(
    "사진 입력 방법",
    ["📷 카메라로 촬영", "🖼️ 사진 파일 선택"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_image = None
if input_method == "📷 카메라로 촬영":
    uploaded_image = st.camera_input(
        "카메라를 열고 촬영 버튼을 눌러주세요",
        help="모바일 브라우저에서 카메라 접근을 허용해 주세요.",
    )
else:
    uploaded_image = st.file_uploader(
        "JPG, JPEG, PNG 형식의 사진을 선택하세요",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

if uploaded_image is None:
    st.info("사진을 촬영하거나 선택하면 여기서 분석 결과를 확인할 수 있습니다.")
else:
    original_image = Image.open(uploaded_image).convert("RGB")

    if not model_path.strip():
        st.error("모델 경로를 입력해 주세요.")
        st.stop()

    st.markdown('<div class="section-label">02 / DETECTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">분석 결과</div>', unsafe_allow_html=True)
    with st.spinner("모델을 준비하고 사진을 분석하고 있어요..."):
        model = load_model(model_path.strip())
        predictions = model.predict(
            source=original_image,
            conf=confidence,
            imgsz=image_size,
            verbose=False,
        )

    result = predictions[0]
    annotated_image = result.plot()[:, :, ::-1]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        class_ids = []
        confidences = []
        st.info("선택한 신뢰도 기준으로 탐지된 사물이 없습니다. 신뢰도를 낮춰 다시 확인해 보세요.")
    else:
        class_ids = boxes.cls.int().tolist()
        confidences = boxes.conf.tolist()

    image_column, summary_column = st.columns([1.55, 1], gap="large")
    with image_column:
        st.image(annotated_image, caption="YOLO 탐지 결과", use_container_width=True)
    with summary_column:
        st.markdown("#### 탐지 요약")
        object_count = len(class_ids)
        average_confidence = (
            sum(confidences) / len(confidences) if confidences else 0
        )
        metric_column1, metric_column2 = st.columns(2)
        metric_column1.metric("찾은 사물", f"{object_count}개")
        metric_column2.metric("평균 신뢰도", f"{average_confidence:.0%}")

        if class_ids:
            class_counts = Counter(result.names[class_id] for class_id in class_ids)
            st.markdown("**감지된 클래스**")
            for class_name, count in class_counts.most_common():
                st.markdown(f"- {class_name} **{count}개**")

            detection_rows = [
                {
                    "사물": result.names[class_id],
                    "신뢰도": f"{score:.1%}",
                }
                for class_id, score in zip(class_ids, confidences)
            ]
            with st.expander("탐지별 신뢰도 보기"):
                st.dataframe(
                    detection_rows,
                    hide_index=True,
                    use_container_width=True,
                )

        image_buffer = BytesIO()
        Image.fromarray(annotated_image).save(image_buffer, format="JPEG", quality=92)
        st.download_button(
            "결과 이미지 저장",
            data=image_buffer.getvalue(),
            file_name="yolo_result.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

st.divider()
st.caption("YOLO Vision · Ultralytics 모델 기반 이미지 객체 탐지")
