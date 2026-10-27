# Works locally (docker run -p 7860:7860 ...) and as a Hugging Face Docker Space.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    CHATBOT_DB=/tmp/chatbot.db

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the Hugging Face models into the image so the first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering as M; AutoTokenizer.from_pretrained('deepset/roberta-base-squad2'); M.from_pretrained('deepset/roberta-base-squad2')" \
 && chmod -R a+rX /opt/hf-cache

COPY . .

# Download NLTK data into the project and train the intent classifier at build time
RUN python -c "import app.nlp" && python -m app.intent_model && chmod -R a+rwX /app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
