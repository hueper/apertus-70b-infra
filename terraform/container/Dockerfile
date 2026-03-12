ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir fastapi uvicorn httpx && \
    pip install --no-cache-dir \
        git+https://github.com/huggingface/transformers.git \
        accelerate && \
    sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' \
        /usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/configs/ovis.py

COPY serve.py /opt/serve.py
COPY entrypoint.sh /opt/entrypoint.sh
RUN sed -i 's/\r$//' /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
