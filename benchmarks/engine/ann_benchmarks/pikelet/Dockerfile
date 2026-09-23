FROM ann-benchmarks

ARG PIKELET_REV=ba41d1f

RUN apt-get update \
    && apt-get install -y --no-install-recommends git g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pybind11

RUN git clone https://github.com/mcn92/pikelet.git /opt/pikelet \
    && cd /opt/pikelet \
    && git checkout "${PIKELET_REV}" \
    && c++ -O3 -DNDEBUG -std=c++17 -shared -fPIC \
       -DPIKELET_ENABLE_AVX2_SIMD -mavx2 -mfma \
       $(python3 -m pybind11 --includes) \
       -Isrc benchmarks/engine/vibe/pikelet_py.cpp \
       -o "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["platlib"] + "/pikelet_py" + sysconfig.get_config_var("EXT_SUFFIX"))')"

RUN python3 -c "import pikelet_py; assert hasattr(pikelet_py, 'PikeletIndex')"
