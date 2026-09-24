FROM ubuntu:24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG TARGETARCH
ARG MRDOCS_VERSION=2026.9.4
ARG MRDOCS_SHA256=be1fbbc516233fe7cc87f66c735b9dbbd23e4bcfadfda705503daac5b6f42ed8
ARG INTEL_SDE_VERSION=10.13.1-2026-07-28
ARG INTEL_SDE_SHA256=94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50

LABEL org.opencontainers.image.title="xvec C++ toolchain" \
      org.opencontainers.image.description="Containerized C++ development environment for intel/xvec" \
      org.opencontainers.image.source="https://github.com/intel/xvec" \
      org.opencontainers.image.url="https://github.com/intel/xvec" \
      org.opencontainers.image.documentation="https://github.com/intel/xvec/blob/main/docs/development-container.md" \
      org.opencontainers.image.licenses="Apache-2.0 WITH LLVM-exception"

ENV MRDOCS_ROOT=/opt/mrdocs
ENV INTEL_SDE_ROOT=/opt/intel-sde
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PATH="${MRDOCS_ROOT}/bin:${INTEL_SDE_ROOT}:${ONEAPI_ROOT}/compiler/latest/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ONEAPI_ROOT}/compiler/latest/lib:${ONEAPI_ROOT}/compiler/latest/lib/x64:${ONEAPI_ROOT}/compiler/latest/opt/compiler/lib"

RUN target_arch="${TARGETARCH:-$(dpkg --print-architecture)}" \
    && if [[ "${target_arch}" != "amd64" ]]; then \
        echo "This image supports only linux/amd64 because Intel SDE is x86-64-only." >&2; \
        exit 1; \
    fi

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        cmake-curses-gui \
        curl \
        doxygen \
        git \
        gnupg \
        graphviz \
        libboost-test-dev \
        make \
        software-properties-common \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN export NO_PROXY= no_proxy= \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl --fail --show-error --location --retry 3 \
        https://apt.llvm.org/llvm-snapshot.gpg.key \
        -o /tmp/llvm.key \
    && gpg --batch --yes --dearmor \
        -o /etc/apt/keyrings/llvm.gpg \
        /tmp/llvm.key \
    && echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] https://apt.llvm.org/noble/ llvm-toolchain-noble-20 main" \
        > /etc/apt/sources.list.d/llvm.list \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && curl --fail --show-error --location --retry 3 \
        https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        -o /tmp/intel-oneapi.key \
    && gpg --batch --yes --dearmor \
        -o /etc/apt/keyrings/oneapi-archive-keyring.gpg \
        /tmp/intel-oneapi.key \
    && echo "deb [signed-by=/etc/apt/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneapi.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clang-20 \
        clangd-20 \
        clang-format-20 \
        clang-tidy-20 \
        gcc-16 \
        g++-16 \
        intel-oneapi-compiler-dpcpp-cpp \
        lldb-20 \
    && test -x /usr/bin/clang++-20 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-16 160 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-16 160 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-16 160 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-16 160 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 200 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 200 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 200 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-20 200 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 200 \
    && update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-20 200 \
    && test -x /opt/intel/oneapi/compiler/latest/bin/icx \
    && test -x /opt/intel/oneapi/compiler/latest/bin/icpx \
    && test -f /opt/intel/oneapi/compiler/latest/env/vars.sh \
    && ln -sf /opt/intel/oneapi/compiler/latest/env/vars.sh /etc/profile.d/20-intel-oneapi-compiler.sh \
    && printf '%s\n' '#!/bin/bash' \
        'source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null' \
        'exec /opt/intel/oneapi/compiler/latest/bin/icx "$@"' \
        > /usr/local/bin/icx \
    && printf '%s\n' '#!/bin/bash' \
        'source /opt/intel/oneapi/compiler/latest/env/vars.sh >/dev/null' \
        'exec /opt/intel/oneapi/compiler/latest/bin/icpx "$@"' \
        > /usr/local/bin/icpx \
    && chmod 0755 /usr/local/bin/icx /usr/local/bin/icpx \
    && rm -f /tmp/llvm.key /tmp/intel-oneapi.key \
    && rm -rf /var/lib/apt/lists/*

RUN export NO_PROXY= no_proxy= \
    && install -d -m 0755 "${MRDOCS_ROOT}" \
    && curl --fail --show-error --location --retry 3 \
        "https://github.com/cppalliance/mrdocs/releases/download/${MRDOCS_VERSION}/MrDocs-${MRDOCS_VERSION}-Linux.tar.gz" \
        -o /tmp/mrdocs.tar.gz \
    && echo "${MRDOCS_SHA256}  /tmp/mrdocs.tar.gz" | sha256sum --check \
    && tar --extract --gzip \
        --file=/tmp/mrdocs.tar.gz \
        --directory="${MRDOCS_ROOT}" \
        --strip-components=1 \
    && rm -f /tmp/mrdocs.tar.gz

RUN export NO_PROXY= no_proxy= \
    && if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then \
        echo "Intel SDE requires an x86-64/amd64 image." >&2; \
        exit 1; \
    fi \
    && curl --fail --show-error --location --retry 3 \
        "https://downloadmirror.intel.com/924984/sde-external-${INTEL_SDE_VERSION}-lin.tar.xz" \
        -o /tmp/intel-sde.tar.xz \
    && echo "${INTEL_SDE_SHA256}  /tmp/intel-sde.tar.xz" | sha256sum --check \
    && install -d -m 0755 "${INTEL_SDE_ROOT}" \
    && tar --extract --xz \
        --file=/tmp/intel-sde.tar.xz \
        --directory="${INTEL_SDE_ROOT}" \
        --strip-components=1 \
    && rm -f /tmp/intel-sde.tar.xz \
    && test -x "${INTEL_SDE_ROOT}/sde64" \
    && test -x "${INTEL_SDE_ROOT}/xed64" \
    && find "${INTEL_SDE_ROOT}" -maxdepth 2 -type f \
        \( -iname 'LICENSE*' -o -iname 'third-party-programs.txt' \) \
        | grep -q . \
    && ln -sf "${INTEL_SDE_ROOT}/sde64" /usr/local/bin/sde \
    && ln -sf "${INTEL_SDE_ROOT}/sde64" /usr/local/bin/sde64 \
    && ln -sf "${INTEL_SDE_ROOT}/xed64" /usr/local/bin/xed \
    && ln -sf "${INTEL_SDE_ROOT}/xed64" /usr/local/bin/xed64

RUN cat <<'EOF' >/tmp/boost_test.cpp
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE xvec_boost_test_smoke
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(smoke)
{
  BOOST_TEST(true);
}
EOF

RUN cat <<'EOF' >/tmp/icpx_smoke.cpp
#include <omp.h>

int main()
{
  int value = 0;
#pragma omp parallel reduction(+:value)
  value += 1;
  return value > 0 ? 0 : 1;
}
EOF

RUN test -x "$(command -v gcc)" \
    && test -x "$(command -v g++)" \
    && test -x "$(command -v cc)" \
    && test -x "$(command -v c++)" \
    && test -x "$(command -v clang)" \
    && test -x "$(command -v clang++)" \
    && test -x "$(command -v clangd)" \
    && test -x "$(command -v clang-format)" \
    && test -x "$(command -v clang-tidy)" \
    && test -x "$(command -v lldb)" \
    && test -x "$(command -v icx)" \
    && test -x "$(command -v icpx)" \
    && cmake --version >/dev/null \
    && ccmake --version >/dev/null \
    && make --version >/dev/null \
    && git --version >/dev/null \
    && curl --version >/dev/null \
    && doxygen --version >/dev/null \
    && dot -V >/dev/null 2>&1 \
    && mrdocs --version >/dev/null \
    && g++ -std=c++20 /tmp/boost_test.cpp -lboost_unit_test_framework -o /tmp/boost_test \
    && /tmp/boost_test --log_level=test_suite \
    && icpx -std=c++20 -fopenmp /tmp/icpx_smoke.cpp -o /tmp/icpx_smoke \
    && /tmp/icpx_smoke \
    && test -x "$(command -v sde)" \
    && test -x "$(command -v sde64)" \
    && test -x "$(command -v xed)" \
    && test -x "$(command -v xed64)" \
    && find "${INTEL_SDE_ROOT}" -maxdepth 2 -type f \
        \( -iname 'LICENSE*' -o -iname 'third-party-programs.txt' \) \
        | grep -q . \
    && test "$(gcc -dumpversion | cut -d. -f1)" = "16" \
    && test "$(clang -dumpversion | cut -d. -f1)" = "20" \
    && rm -f /tmp/boost_test.cpp /tmp/boost_test /tmp/icpx_smoke.cpp /tmp/icpx_smoke

CMD ["/bin/bash"]
