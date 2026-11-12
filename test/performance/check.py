#!/usr/bin/env python3
"""Compare the llvm-mca cycle count of each scenario's kernel against stored limits.

The analysed assembly is written to build-performance/<scenario>.<target>.s,
and the full llvm-mca report to <scenario>.<target>.mca.txt.
Run with --help for usage.
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "baselines.json"
ASM_DIR = HERE.parents[1] / "build-performance"  # kernels to review; git-ignored

# name: (-march, llvm-mca cpu, extra defines)
TARGETS = {
    "sse": ("goldmont", "goldmont", []),
    "avx": ("core-avx-i", "ivybridge", []),
    "avx2": ("core-avx2", "haswell", []),
    "avx512": ("skylake-avx512", "skylake-avx512", []),
    "avxsnc": ("icelake-server", "icelake-server", []),
    "avxglc": ("sapphirerapids", "sapphirerapids", []),
}


def run(*cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} failed:\n{result.stderr.strip()}")
    return result.stdout


def model(source, target, march, cpu, defines):
    """Return the modeled cycles for one execution of the kernel, or None if it contains a call or branch."""
    cxx = shlex.split(os.environ.get("CXX", "clang++"))
    flags = shlex.split(os.environ.get("CXXFLAGS", ""))
    name = "xvec_" + source.stem
    asm = run(
        *cxx, "--target=x86_64-pc-linux-gnu", "-std=c++20", "-O3", "-Wno-psabi", "-Wno-return-type-c-linkage",
        "-march=" + march, *("-D" + d for d in defines), *flags,
        "-I", str(HERE.parents[1] / "include"), "-S", str(source), "-o", "-",
    )
    match = re.search(rf"^{name}:[^\n]*\n(.*?)^\s*\.size\s+{name},", asm, re.M | re.S)
    if not match:
        raise RuntimeError(f"{name} not found in assembly")
    body = match[1]
    kernel = "\n".join(
        line.split("#")[0] for line in body.splitlines()
        if not line.strip().startswith((".", "ret", "#")) and not line.strip().endswith(":")
    )
    ASM_DIR.mkdir(exist_ok=True)
    (ASM_DIR / f"{source.stem}.{target}.s").write_text(kernel + "\n")
    if re.search(r"^\s*(call|j[a-z]+)\b", kernel, re.M):
        return None  # a call or branch makes the straight-line cycle model meaningless
    if not kernel.strip():
        return 0  # nothing to execute; llvm-mca rejects empty input
    with tempfile.NamedTemporaryFile("w", suffix=".s") as f:
        f.write(kernel)
        f.flush()
        mca = [
            os.environ.get("CXX", "clang++").replace("clang++", "llvm-mca"),
            "-mtriple=x86_64-pc-linux-gnu", "-mcpu=" + cpu, "-iterations=1",
        ]
        report = run(*mca, "-json", f.name)
        (ASM_DIR / f"{source.stem}.{target}.mca.txt").write_text(
            run(*mca, "-instruction-info", "-resource-pressure", f.name)
        )
    summary = json.loads(report)["CodeRegions"][0]["SummaryView"]
    return summary["TotalCycles"]


def main():
    parser = argparse.ArgumentParser(
        description="Compare the llvm-mca cycle count of each scenario against baselines.json.",
        epilog="Environment: CXX (default clang++; llvm-mca is derived from it), CXXFLAGS.",
    )
    parser.add_argument("scenarios", nargs="*", metavar="SCENARIO",
                        help="scenario names to run (default: all)")
    parser.add_argument("--update", action="store_true",
                        help="record the current cycle counts as the new baselines")
    args = parser.parse_args()

    sources = [
        p for p in sorted((HERE / "scenarios").glob("*.cpp"))
        if not args.scenarios or p.stem in args.scenarios
    ]
    if len(sources) != (len(set(args.scenarios)) or len(sources)):
        parser.error("unknown scenario name")

    baselines = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    failed = False
    for source in sources:
        for target, (march, cpu, defines) in TARGETS.items():
            try:
                actual = model(source, target, march, cpu, defines)
            except RuntimeError as error:
                failed = True
                print(f"FAIL {source.stem} {target}: {error}")
                continue
            recorded = target in baselines.get(source.stem, {})
            limit = baselines.get(source.stem, {}).get(target)
            name = f"{source.stem} {target}"
            if args.update:
                baselines.setdefault(source.stem, {})[target] = actual
                print(f"set  {name}: " + ("call/branch, not modeled" if actual is None else f"{actual} cycles"))
            elif not recorded:
                failed = True
                print(f"FAIL {name}: no baseline")
            elif actual is None:
                if limit is None:
                    print(f"skip {name}: contains a call/branch")
                else:
                    failed = True
                    print(f"FAIL {name}: now contains a call/branch (was {limit} cycles)")
            elif limit is None:
                print(f"ok   {name}: {actual} cycles, now straight-line; run --update to track it")
            else:
                bad = actual > limit
                failed |= bad
                print(f"{'FAIL' if bad else 'ok  '} {name}: {actual} cycles, limit {limit}")
    if args.update:
        BASELINES.write_text(json.dumps(baselines, indent=1, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
