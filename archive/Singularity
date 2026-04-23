Bootstrap: docker
From: nvcr.io/nvidia/pytorch:22.04-py3

%runscript
 
    "$@"

%post

    apt-get -y update
    pip3 install jupyterlab
    pip3 install ipywidgets
    pip3 install torch_tb_profiler
    
    # Read https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
    apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
    wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-systems-2022.1.1 && \
     rm -rf /var/lib/apt/lists/*

%files

    #../English/* /workspace/

%environment
    export XDG_RUNTIME_DIR=
    export LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.8/dist-packages:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export PATH="/opt/nvidia/nsight-systems/2022.2.1/bin:/usr/local/bin:/bin:/usr/local/cuda/bin:/usr/bin${PATH:+:${PATH}}"

%labels

AUTHOR Tosin
