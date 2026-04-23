# Profiling AI Software Bootcamp

The Profiling AI Software Bootcamp covers the process and tools for profiling AI and machine learning applications to fully utilize high-performance systems. Attendees will learn to profile applications using NVIDIA Nsight™ Systems, a system-wide performance analysis tool; analyze and identify optimization opportunities; and improve application performance to scale efficiently across systems of any size and number of CPUs and GPUs. Additionally, this bootcamp will walk through the system topology to learn the dynamics of FP8 precision, multi-GPU, and multi-node connections and architecture.

## Deploying the Labs

### Prerequisites

To run this tutorial, you will need a DGX machine with a minimum of NVIDIA Hopper GPU Architecture (H100).

- You need a Linux Machine.
- Install latest [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install). 
- Install the latest [Docker](https://docs.docker.com/engine/install/) or [Singularity](https://sylabs.io/docs/).
- Install NVIDIA [Nsight Systems](https://developer.nvidia.com/nsight-systems).
- Install [UV](https://docs.astral.sh/uv/getting-started/installation/)

### Tested environment

We tested and ran all labs on a DGX machine equipped with a H100 GPUs (80GB).


### Deploying Lab 1, 2, & 3

#### Please follow the commands below:

```bash

#create uv venv 

uv venv --python 3.12

#activate the venv

source .venv/bin/activate

# Install the dependencies.

cd ~/Profiling-AI-Software-Bootcamp

uv pip install -r requirements.txt

```
#### Downloading Dataset

```bash

# navigate to workspace directory

cd ~/Profiling-AI-Software-Bootcamp/workspace

# run the download script

uv run python source_code/download-data.py

# unzip the data

unzip -u data/data-list.zip -d data/

unzip -u source_code/saved_models.zip -d source_code/

```

#### Running the Jupyter Notebook

```bash
# Please ensure that venv is activated otherwise activate it by running: source .venv/bin/activate

jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=./workspace

```

 
### Deploying Lab 4 with container 

You can deploy this material using Conda, Docker or Apptainer containers. Please refer to the respective sections for the instructions.


#### Running Docker Container 

To run the Labs, you will need access to 2 nodes(at least 4 GPUs per node). Build a Docker container by following these steps:  

- Open a terminal window and navigate to the directory where `Dockerfile` file is located (`cd ~/Profiling-AI-Software-Bootcamp`)
- To build the docker container, run : `sudo docker build -f Dockerfile --network=host -t <imagename>:<tagnumber> .`, for instance: 


```bash

sudo docker build -f Dockerfile --network=host -t tecont:v1 .

```
- To run the built container : 

```bash

docker run --rm -it --gpus all -p 8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ./workspace:/workspace tecont:v1 jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```


flags:
- `--rm` will delete the container when finished.
- `-it` means run in interactive mode.
- `--gpus` option makes GPUs accessible inside the container.
- `-v` is used to mount host directories in the container filesystem.
- `--network=host` will share the host’s network stack to the container.
- `-p` flag explicitly maps a single port or range of ports.


Open the browser at `http://localhost:8888` and click on the `start_here.ipynb`. Go to the table of content and click on Lab 1: `Preprocessing Multi-turn Conversational Dataset`.
As soon as you are done with the rest of the labs, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl d` in the terminal window.



#### Running Singularity Container


- Build the Labs Singularity container with: 

```bash

apptainer build --fakeroot --sandbox tecont.simg Singularity


```

- To run the built container: 

```bash

singularity run --nv -B workspace:/workspace tecont.simg jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```
 
 The `-B` flag mounts local directories in the container filesystem and ensures changes are stored locally in the project folder. Open jupyter lab in the browser: http://localhost:8888

You may start working on the labs by clicking the `start_here.ipynb` notebook.

When you finish these notebooks, shut down jupyter lab by selecting `File > Shut Down` in the top left corner, then shut down the Singularity container by typing `exit` or pressing `ctrl + d` in the terminal window.




## Known issues

- invalid device ordinal


```python

W0418 12:45:25.804000 704756 torch/distributed/run.py:852] 
W0418 12:45:25.804000 704756 torch/distributed/run.py:852] *****************************************
W0418 12:45:25.804000 704756 torch/distributed/run.py:852] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
W0418 12:45:25.804000 704756 torch/distributed/run.py:852] *****************************************
Local rank  2
Local rank  0
Local rank  3
Local rank  1

[2026-04-18 12:45:41] gpu002:704798:704908 [0] transport/p2p.cc:275 NCCL WARN Cuda failure 101 'invalid device ordinal'

[2026-04-18 12:45:41] gpu002:704798:704904 [0] proxy.cc:1632 NCCL WARN [Service thread] Accept failed Resource temporarily unavailable

[2026-04-18 12:45:41] gpu002:704800:704915 [1] misc/socket.cc:72 NCCL WARN socketProgress: Connection closed by remote peer gpu002.cm.cluster<44189>

[2026-04-18 12:45:41] gpu002:704800:704915 [1] proxy.cc:1208 NCCL WARN ncclProxyClientGetFd call to tpRank 4 handle 0x15518002cbb0 failed : 6
[rank4]: Traceback (most recent call last):

```

The above issues may be raised when running the multinode Notebook. This is because 4 GPUs are required for each node, resulting in a total of 8 GPUs. If the Slurm manager allocates all 8 GPUs from a single node, an invalid device ordinal issue is raised.
This can be avoided by ensuring that 4 of the GPUs are allocated from a different node. Alternatively, you can target specific nodes in the Slurm script using the nodelist flag (e.g., #SBATCH --nodelist=gpu004,gpu005).
