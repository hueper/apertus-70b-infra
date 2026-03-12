ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    "transformers>=4.49.0" \
    accelerate

COPY serve.py /opt/serve.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN sed -i 's/\r$//' /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
