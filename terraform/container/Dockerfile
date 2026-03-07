ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Pin transformers below the version that conflicts with vLLM 0.9.0's ovis.py
# (aimv2 double-registration error introduced in transformers >= 4.54.0)
RUN pip install --no-cache-dir "transformers<4.54.0"

RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY serve.py /opt/serve.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN sed -i 's/\r$//' /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
