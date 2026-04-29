# check=skip=UndefinedVar
# ==============================================================================
# Themis Training Container (Single-Stage)
# ==============================================================================
#
# Purpose:
#   Builds a Docker image optimised for multi-node, multi-GPU reward model
#   training on AWS p5 (H100) and p4d (A100) instances with EFA networking.
#
# Base image:
#   NVIDIA PyTorch NGC container (nvcr.io/nvidia/pytorch:26.04-py3) which
#   includes PyTorch, CUDA toolkit, cuDNN, NCCL, and Triton.
#
# What this Dockerfile adds on top of the base:
#   1. AWS Elastic Fabric Adapter (EFA) installer — kernel-bypass networking
#      for NCCL all-reduce at 400+ Gbps between nodes.
#   2. NVIDIA GDRCopy — enables GPU-direct RDMA for lowest-latency NCCL
#      transfers over EFA.
#   3. AWS-OFI-NCCL plugin — bridges NCCL collective operations to the
#      libfabric transport layer used by EFA.
#   4. Open MPI — configured with EFA-compatible settings (disabling UCX/IB
#      transports that conflict with EFA).
#   5. Python packages: HuggingFace Transformers, Accelerate, DeepSpeed,
#      Flash Attention (via flash-linear-attention), Liger Kernel, W&B.
#
# Why single-stage:
#   AWS-OFI-NCCL links against libfabric from the EFA installer at compile time.
#   A multi-stage build copies the compiled artifact into a runtime stage that
#   has a separate EFA install, causing libfabric version/path mismatches and
#   NCCL "No eligible providers" errors. Building everything in one stage
#   ensures all libraries link against the same EFA/libfabric installation.
#
# Size optimisations:
#   - All apt installs, pip installs, and cleanups happen within single RUN
#     commands to avoid layer bloat from intermediate files.
#   - Shared libraries are stripped of debug symbols.
#   - Python bytecache and pip wheel caches are purged.
#   - Build dependencies for GDRCopy/AWS-OFI-NCCL are removed after compilation.
#
# Build:
#   docker build -f Themis.Dockerfile -t ineil77/themis:29042026-3 .
#
# Usage with enroot:
#   enroot import docker://ineil77/themis:29042026-3
#   enroot create --name AWS_Themis ineil77+themis+29042026-3.sqsh
#   enroot start --root --rw --mount ... AWS_Themis <command>
#
# Architecture support:
#   ARG TORCH_CUDA_ARCH_LIST covers: A100 (8.0), A6000 (8.6), L40/4090 (8.9),
#   H100 (9.0), B200 (10.0), and future GPUs via PTX (10.3+PTX).
#
# ==============================================================================

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

FROM nvcr.io/nvidia/pytorch:26.04-py3

ARG AWS_OFI_NCCL_VERSION=v1.19.0
ARG DEEPSPEED_VERSION=0.18.9
ARG EFA_INSTALLER_VERSION=1.48.0
ARG GDRCOPY_VERSION=v2.5.2
ARG MAX_JOBS=160
ARG OPEN_MPI_PATH=/opt/amazon/openmpi
ARG TILELANG_VERSION=0.1.9
ARG TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 10.3+PTX"
ARG TRANSFORMERS_VERSION=5.6.2

# ---- System preparation ----
# Remove IB libraries that conflict with EFA, install all needed packages
# (both build-time and runtime). Build-only packages are removed later.
RUN apt-get update -y && apt-get upgrade -y \
    && apt-get remove -y --allow-change-held-packages \
        ibverbs-utils \
        libibverbs-dev \
        libibverbs1 \
        libmlx5-1 \
    && rm -rf /opt/hpcx/ompi /usr/local/mpi /usr/local/ucx \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils \
        autoconf \
        automake \
        build-essential \
        cmake \
        curl \
        environment-modules \
        gdb \
        git \
        kmod \
        libhwloc-dev \
        libnuma-dev \
        libaio-dev \
        libtool \
        ninja-build \
        openssh-client \
        openssh-server \
        tcl \
        vim \
    && apt-get remove -y python3-blinker \
    && apt-get autoremove -y \
    && ldconfig

# ---- SSH for MPI multi-node launchers ----
RUN mkdir -p /var/run/sshd \
    && sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config \
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    && rm -rf /root/.ssh/ \
    && mkdir -p /root/.ssh/ \
    && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
    && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
    && printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# ---- EFA Installer ----
# Installs libfabric, NCCL EFA plugin (libnccl-env.so, libnccl-net.so), and
# Open MPI into /opt/amazon/. Must come before AWS-OFI-NCCL which links
# against the libfabric installed here.
RUN cd /tmp \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/aws-efa-installer /tmp/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz

# ---- GDRCopy ----
RUN git clone --depth 1 -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install \
    && rm -rf /tmp/gdrcopy

# ---- AWS-OFI-NCCL ----
# Links against the EFA libfabric and NCCL installed above — must be built
# in-place (not in a separate stage) to avoid library version mismatches.
SHELL ["/bin/bash", "-c"]
RUN cd /tmp \
    && curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && rm -rf /tmp/aws-ofi-nccl-*
SHELL ["/bin/sh", "-c"]

# ---- Clean apt lists (no further apt-get calls after this point) ----
RUN rm -rf /var/lib/apt/lists/*

# ---- Environment paths ----
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:/opt/gdrcopy/lib:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:/usr/local/cuda/compat/:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:$CPATH
ENV PATH=/opt/gdrcopy/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

# ---- Open MPI configuration ----
RUN echo "hwloc_base_binding_policy = none" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf \
    && echo "rmaps_base_mapping_policy = slot" >> /opt/amazon/openmpi/etc/openmpi-mca-params.conf \
    && mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
    && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
    && echo '/opt/amazon/openmpi/bin/mpirun.real "$@"' >> $OPEN_MPI_PATH/bin/mpirun \
    && chmod a+x $OPEN_MPI_PATH/bin/mpirun

ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth_def_agent \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PMIX_MCA_gds=hash

# ---- Python packages ----
RUN pip3 install --no-cache-dir \
        accelerate \
        annotated-types \
        awscli \
        datasets \
        deepspeed-kernels \
        einops \
        editables \
        hjson \
        msgpack \
        ninja \
        nvitop \
        oneccl-devel \
        packaging \
        psutil \
        py-cpuinfo \
        pydantic-core \
        pydantic \
        pynvml \
        python-etcd \
        tqdm \
        transformers==${TRANSFORMERS_VERSION} \
        wandb \
    && DS_BUILD_OPS=1 \
       DS_BUILD_SPARSE_ATTN=0 \
       DS_BUILD_STOCHASTIC_TRANSFORMER=0 \
       DS_BUILD_EVOFORMER_ATTN=0 \
       pip install --no-cache-dir --no-build-isolation deepspeed==${DEEPSPEED_VERSION} \
           --config-settings "--global-option=build_ext" \
           --config-settings "--global-option=-j128" \
    && pip3 install --no-cache-dir --no-build-isolation \
        causal-conv1d \
        flash-linear-attention \
        liger-kernel-nightly \
        tilelang==${TILELANG_VERSION} \
    && rm -rf /root/.cache/pip /tmp/pip-* /tmp/*.whl /tmp/*.tar.gz \
    && find /usr/local/lib/python3* -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3* -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3* -name "*.pyc" -delete 2>/dev/null || true

# ---- Strip debug symbols from shared libraries (saves ~200-500MB) ----
RUN find /opt/gdrcopy /opt/amazon /opt/aws-ofi-nccl \
        -name "*.so*" -exec strip --strip-debug {} \; 2>/dev/null || true \
    && find /usr/local/lib/python3*/dist-packages \
        -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true

# ---- Remove build-only dependencies to save space ----
RUN apt-get update -y \
    && apt-get remove -y --autoremove \
        autoconf \
        automake \
        libtool \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
