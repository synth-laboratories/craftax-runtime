ARG SCORER_BASE_IMAGE
ARG SCORER_BUILDER_IMAGE

FROM ${SCORER_BUILDER_IMAGE} AS rust-builder

WORKDIR /build
COPY gold_rust /build/gold_rust
RUN CARGO_TARGET_DIR=/build/target cargo build \
    --release \
    --locked \
    --quiet \
    --manifest-path /build/gold_rust/Cargo.toml \
    --bin craftax_repl

FROM ${SCORER_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONNOUSERSITE=1
ENV PYTHONUNBUFFERED=1

LABEL ai.synth.gamebench.runtime.security-context="seccomp=unconfined,cap-drop=ALL,no-new-privileges"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY containers/scorer_service/requirements.txt /tmp/scorer-requirements.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/scorer-requirements.txt

COPY . /app
ENV PYTHONPATH=/app

COPY --from=rust-builder \
    /build/target/release/craftax_repl \
    /app/gold_rust/target/gamebench-host/linux-x86_64/release/craftax_repl
RUN python -c "from containers.codepolicy.rust_repl_session import ensure_rust_repl_binary; ensure_rust_repl_binary()"

USER 65532:65532
ENTRYPOINT ["python", "-m", "containers.scorer_service.service"]
