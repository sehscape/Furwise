# Furwise demo image, sized for a free Hugging Face Space (CPU basic, 16 GB RAM).
#   docker build -t furwise .
#   docker run -p 7860:7860 -e GROQ_API_KEY=... furwise
# The corpus and vector index (data/processed, data/index) are built locally with
# `furwise fetch && furwise build && furwise index` and copied in at deploy time.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .
RUN pip install -e . --no-deps

USER user
# Download the embedding and reranker models at build time, not on the first request.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
