FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install jupyterlab==4.5.1 nsight-python==0.9.5

# Start JupyterLab
CMD ["jupyter-lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--NotebookApp.token=", "--notebook-dir=/workspace-aiprofiler/workspace"]