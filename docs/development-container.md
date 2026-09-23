# Development container and toolchain image

The repository includes a containerized C++ development environment for xvec.
The image definition lives in the repository `Dockerfile`, and GitHub Actions can
publish the resulting image as `ghcr.io/intel/xvec-toolchain:latest`.

## Build the image with Docker

### Behind the corporate proxy

Use the tested build command below when external LLVM, Intel, and GitHub
downloads must go through a corporate proxy:

```bash
sudo docker build --no-cache --progress=plain \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-$http_proxy}" \
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}" \
  --build-arg http_proxy="${http_proxy:-$HTTP_PROXY}" \
  --build-arg https_proxy="${https_proxy:-$HTTPS_PROXY}" \
  --build-arg NO_PROXY= \
  --build-arg no_proxy= \
  -t xvec-toolchain:latest .
```

Both uppercase and lowercase variables are supplied because different tools and
build environments consult different spellings. Both `NO_PROXY` and `no_proxy`
are explicitly cleared so the external LLVM, Intel, and GitHub hosts do not
bypass the proxy unexpectedly. Do not bake proxy values into the image with
`ENV`; keep them as per-build inputs.

### Without a proxy

For normal internet-connected systems, a simpler build is sufficient:

```bash
sudo docker build --progress=plain -t xvec-toolchain:latest .
```

## Run Docker manually

Bind-mount the checkout so the container can access the source tree:

```bash
sudo docker run --rm -it \
  --mount type=bind,source="$PWD",target=/workspace \
  --workdir /workspace \
  xvec-toolchain:latest
```

Intel SDE is already installed in the image. The extra runtime options below are
permissions for SDE and Pin-based tracing, not installation steps:

```bash
sudo docker run --rm -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --mount type=bind,source="$PWD",target=/workspace \
  --workdir /workspace \
  xvec-toolchain:latest
```

Example SDE invocation:

```bash
sde64 -spr -- ./my-program
```

## Use the toolchain image in VS Code locally

1. Install Docker and Visual Studio Code.
2. Install the Dev Containers extension
   (`ms-vscode-remote.remote-containers`).
3. Clone this repository locally and open the repository folder in VS Code.
4. If the GHCR package is private, authenticate before reopening in the
   container. For example:

   ```bash
   gh auth token | docker login ghcr.io --username USERNAME --password-stdin
   ```

5. Run **Dev Containers: Reopen in Container** from the Command Palette.

When VS Code opens the dev container, it automatically bind-mounts the
repository workspace. You do not need to add a manual Docker `--mount` option in
that workflow. The first image pull can be large, but subsequent container
starts reuse the cached image and layers that already exist on the local system.

After changes to the image or dev-container definition, use **Dev Containers:
Rebuild Container** or **Reopen in Container** again to pick up the update.

The default build directory is `build/`. Configure and build with CMake or
`ccmake`:

```bash
cmake -S test -B build -DCMAKE_CXX_COMPILER="${CXX:-g++}"
cmake --build build --parallel
```

To adjust cache entries interactively, run `ccmake -S test -B build`.

To choose a different compiler, set `CC` and `CXX` before configuring a fresh
build directory. Typical examples are `clang`/`clang++` or `icx`/`icpx`.

The image is built only for `linux/amd64` because Intel SDE is distributed for
x86-64 Linux. Codespaces can consume the same dev-container definition, but SDE
or Pin-based tracing may still be limited by nested or otherwise restricted
runtime environments even though the binaries are installed in the image.

## GHCR image information

The published image name is:

```text
ghcr.io/intel/xvec-toolchain:latest
```

The `.github/workflows/toolchain-image.yml` workflow validates pull requests by
building the image without publishing it. Pushes to `main` publish `latest`,
branch, and commit-SHA tags through GHCR. Access to the package depends on the
repository or package visibility settings; users may need to authenticate to
pull the image when the package is private.

## Intel SDE redistribution note

The image bundles Intel SDE for development and test workflows, and the
extracted SDE kit preserves Intel's bundled `Licenses/` materials inside the
image. A mirrored copy of Intel's bundled SDE license text indicates that
redistribution of the unmodified package may be allowed when the notices and
terms are reproduced, but that conclusion has not been re-verified against the
exact pinned `10.13.1-2026-07-28` archive from Intel in this repository
workflow. Keep the GHCR package private until Intel or your legal team confirms
that the intended public redistribution is acceptable for the pinned release.
