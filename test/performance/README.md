# Modeled performance tests

Each `scenarios/<name>.cpp` defines an `extern "C"` function `xvec_<name>`.
`check.py` compiles it for every target in `TARGETS`, runs `llvm-mca` on the
generated assembly and fails if the modeled cycle count for one execution of the
kernel (`-iterations=1`) exceeds the limit in `baselines.json`.

    python3 test/performance/check.py                       # check everything
    python3 test/performance/check.py popcount_u32x16       # one scenario
    python3 test/performance/check.py --update [SCENARIO]   # accept current numbers

The assembly given to `llvm-mca` is written to `build-performance/<scenario>.<target>.s`.
The full `llvm-mca` report is next to it as `.mca.txt`. Review them, and if it is correct run `--update` to accept the reported numbers as the new baseline.

Set `CXX` (e.g. `clang++-18`) to select the toolchain; `llvm-mca` is derived from the `clang++` word in it (`llvm-mca-18`), so a wrapper such as `CXX='ccache clang++-18'` works. Set `LLVM_MCA` to override. `CXXFLAGS` adds options. Baselines depend
on the compiler and llvm-mca versions, so regenerate them whenever you change either.
A kernel containing a call or branch can't be modeled as straight-line code, so it is
recorded as `null` in `baselines.json` and skipped. It fails only if a previously
modeled kernel starts containing a call or branch.
