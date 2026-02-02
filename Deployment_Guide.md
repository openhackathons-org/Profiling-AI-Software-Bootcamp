# Profiling AI Software Bootcamp

## Prerequisites

To run this bootcamp you will need a machine with NVIDIA GPUs. The profiling tools require:

- **GPU**: NVIDIA GPUs with Ampere architecture and above (SM 80+) for Nsight Systems and Nsight Compute.
- **Container Runtime**: Install [Docker](https://docs.docker.com/get-docker/) or [Singularity](https://sylabs.io/docs/)
- **NVIDIA Toolkit**: Install NVIDIA toolkit, [Nsight Systems](https://developer.nvidia.com/nsight-systems).
- **NGC Account**: Building the base container image requires users to create a [NGC account and generate an API key](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account).
- **Linux Machine**: Ubuntu Operating System.

## Tested Environment

We tested and ran all labs on a DGX machine equipped with A100 and H100 GPUs.

## Deploying the Labs

You can deploy this material Docker containers. Please refer to the respective sections for the instructions.

### Running Docker Container
To run the labs, you will need access to a single GPU. Build a Docker container by following these steps:

1. Open a terminal window and navigate to the directory where the Dockerfile is located (e.g., `cd ~/Profiling-AI-Software-Bootcamp`)

2. To build the docker container, run:
```bash
sudo docker build -t aiprofiler-jupyter:latest .
```

3. To run the built container:
```bash
docker run -it --gpus "all" \
    -p 8888:8888 --rm \
    -v /path/to/Profiling-AI-Software-Bootcamp:/workspace-aiprofiler \
    aiprofiler-jupyter:latest
```

**Flag descriptions:**
- `--rm` cleans up temporary images created during the running of the container
- `-it` enables interactive mode and killing the jupyter server with `ctrl-c`
- `--gpus=all` enables all NVIDIA GPUs during container runtime
- `-v` mounts local directories in the container filesystem
- `-p` explicitly maps port 8888

When this command is run, you can browse to the serving machine on port 8888 using any web browser to access the labs. For instance, if running on the local machine, the web browser should be pointed to http://localhost:8888.

4. Once inside the container, open the jupyter lab in browser: http://localhost:8888, and start the lab by clicking on the `start_here.ipynb` notebook.

5. As soon as you are done with the labs, shut down jupyter lab by selecting **File > Shut Down** and exit the container by typing `exit` or pressing `ctrl + d` in the terminal window.

## Troubleshooting

#### Container fails to start or exits immediately

Check the container logs:
  ```bash
  docker logs <container_id>
  ```
Ensure the workspace path in the `-v` flag points to the correct local directory

#### Nsight Systems or Nsight Compute commands not found

The tools should be pre-installed in the container. Verify installation:
  ```bash
  nsys --version
  ```

For additional support, please refer to the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) or open an issue in the repository.

#### ERR_NVGPUCTRPERM: Permission Issue with GPU Performance Counters

If you encounter `ERR_NVGPUCTRPERM` error when profiling, ensure the container is started with `--cap-add=SYS_ADMIN`. For a permanent solution, enable access on the host: `sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'` then reboot. 

See [NVIDIA's solutions guide](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters) for details.
