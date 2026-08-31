FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8104

WORKDIR /app

COPY containers/react/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# The Trace V5 emitter is supplied as the release-approved wheel, never from a
# mutable checkout or registry fallback. The named context must contain the
# canonical artifact at
# /synth_containers-0.4.0.20260725-py3-none-any.whl:
#   docker buildx build \
#     --build-context synth_containers=/path/to/wheel-context \
#     -f containers/react/Dockerfile .
COPY --from=synth_containers /synth_containers-0.4.0.20260725-py3-none-any.whl /tmp/synth_containers-0.4.0.20260725-py3-none-any.whl
RUN echo "9d40f21be7ea21a72aec8e316b4de4524e20cee54cc641bfdb2439413daa22b0  /tmp/synth_containers-0.4.0.20260725-py3-none-any.whl" \
        | sha256sum --check --strict - \
    && python -m pip install --no-cache-dir /tmp/synth_containers-0.4.0.20260725-py3-none-any.whl \
    && python -c "from importlib.metadata import version; assert version('synth-containers') == '0.4.0.20260725'"

LABEL ai.synth.containers.version="0.4.0.20260725"
LABEL ai.synth.containers.source_commit="0456dc5ea4900e54714acc26867345d84a07b9ff"
LABEL ai.synth.containers.wheel_sha256="9d40f21be7ea21a72aec8e316b4de4524e20cee54cc641bfdb2439413daa22b0"

COPY . /app

ENV PYTHONPATH=/app:/app/gold_python:/app/shared

EXPOSE 8104

CMD ["python", "-m", "containers.react.craftax_singleplayer_container"]
