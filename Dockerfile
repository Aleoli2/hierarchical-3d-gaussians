ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.1.1


FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

ARG COLMAP_GIT_COMMIT=main
ARG CUDA_ARCHITECTURES=native
ENV QT_XCB_GL_INTEGRATION=xcb_egl

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        ninja-build \
        build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev

# Build and install COLMAP.
RUN git clone https://github.com/colmap/colmap.git
ARG TORCH_CUDA_ARCH_LIST
RUN cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_INSTALL_PREFIX=/colmap-install && \
    ninja install

FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as runtime

ARG CUDA_ARCHITECTURES=native

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        git \
        cmake \
        build-essential \
        software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && \
    apt-get install -y python3.12 \
        python3-pip \
        python-is-python3 
RUN git clone https://github.com/graphdeco-inria/hierarchical-3d-gaussians.git --recursive
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
ARG TORCH_CUDA_ARCH_LIST
RUN cd hierarchical-3d-gaussians && pip install -r requirements.txt

# Compiling hierarchy generator and merger
RUN cd hierarchical-3d-gaussians/submodules/gaussianhierarchy && \
    cmake . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j --config Release

#Compiling the real-time viwer
RUN apt-get update && \
    apt-get install -y \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \ 
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev

RUN cd hierarchical-3d-gaussians/SIBR_viewers && \
    git clone https://github.com/graphdeco-inria/hierarchy-viewer.git src/projects/hierarchyviewer && \
    cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_IBR_HIERARCHYVIEWER=ON -DBUILD_IBR_ULR=OFF -DBUILD_IBR_DATASET_TOOLS=OFF -DBUILD_IBR_GAUSSIANVIEWER=OFF && \ 
    cmake --build build -j --target install --config Release


# Minimal dependencies to run COLMAP binary compiled in the builder stage.
# Note: this reduces the size of the final image considerably, since all the
# build dependencies are not needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.74.0 \
        libc6 \
        libceres2 \
        libfreeimage3 \
        libgcc-s1 \
        libgl1 \
        libglew2.2 \
        libgoogle-glog0v5 \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5

COPY --from=builder /colmap-install/ /usr/local/
COPY ./submodules/Depth-Anything-V2/checkpoints/ /hierarchical-3d-gaussians/submodules/Depth-Anything-V2/checkpoints/
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,display