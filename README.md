[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

# Profiling AI Software Bootcamp

The Profiling AI Software Bootcamp covers the process and tools for profiling AI and machine learning applications to fully utilize high-performance systems. Attendees will learn to profile applications using NVIDIA Nsight™ Systems, a system-wide performance analysis tool; analyze and identify optimization opportunities; and improve application performance to scale efficiently across systems of any size and number of CPUs and GPUs. Additionally, this bootcamp will walk through the system topology to learn the dynamics of FP8 precision, multi-GPU, and multi-node connections and architecture.

## Bootcamp Content

This content contains an introductory hands-on lab followed by 4 advanced labs:

- **Intro Lab: Introduction to Profiling** (5 notebooks)
  - Profiling Concepts
  - PyTorch Profiler
  - Profiler Exports for Batch Workflows
  - Mixed Precision Training (AMP)
  - Introduction to Nsight Systems
- **Lab 1:** System Topology
- **Lab 2:** Distributed Training Strategy
- **Lab 3:** Performance Overview
- **Lab 4:** Transformer Engine

The Intro Lab uses a hands-on, two-bug demo approach: attendees discover and fix real performance issues using PyTorch Profiler and Nsight Systems, learning when each tool is most effective.


## Bootcamp Duration

The duration of the tutorial is 4 hours 30 minutes.


## Tools and Frameworks

The tools and frameworks used in this bootcamp are as follows:
- [PyTorch](https://pytorch.org/) with [PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)
- [NVIDIA® Nsight™ Systems](https://developer.nvidia.com/nsight-systems)
- [TensorBoard](https://www.tensorflow.org/tensorboard) (for PyTorch Profiler visualization)
- [Perfetto](https://ui.perfetto.dev/) (for viewing exported Chrome traces)
- NVTX annotations for custom profiling ranges


## Deploying the Bootcamp Material

To deploy the Labs, please refer to the deployment guide presented [here](Deployment_Guide.md)

## Attribution

This material originates from the OpenHackathons GitHub repository. Check out additional materials [here](https://github.com/openhackathons-org).

Don't forget to check out additional [Open Hackathons Resources](https://www.openhackathons.org/s/technical-resources) and join our [OpenACC and Hackathons Slack Channel](https://www.openacc.org/community#slack) to share your experience and get more help from the community.


## Licensing

Copyright © 2026 OpenACC-Standard.org. This material is released by OpenACC-Standard.org, in collaboration with NVIDIA Corporation, under the Creative Commons Attribution 4.0 International (CC BY 4.0). These materials may include references to hardware and software developed by other entities; all applicable licensing and copyrights apply.
