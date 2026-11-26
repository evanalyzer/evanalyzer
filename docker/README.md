# EVAnalyzer

Find, classify and measure objects in microscopy images - reproducibly, on
ordinary computers. This image holds EVAnalyzer's **command line** and
**server**, without the desktop app: run analyses on a server or compute
cluster, or serve them to the desktop app on other computers.

Project, downloads and documentation:
[github.com/evanalyzer/evanalyzer](https://github.com/evanalyzer/evanalyzer)

## Tags

| Tag | Platforms |
|---|---|
| `<version>`, `latest` | linux/amd64, linux/arm64 |

`latest` is the newest stable release; pre-releases (e.g. `1.2.0-alpha.1`)
get only their version tag. The AI steps (Cellpose-SAM, StarDist, U-Net)
run on the CPU in the container: the minimal base image has no Vulkan
driver for a GPU.

## Analyze a project

Mount the folder with the project and its images at `/data`:

```sh
docker run --rm -v "$PWD":/data joda001/evanalyzer \
    cli analyze --project /data/my.evaproj --images /data/images --job-name run1
```

`--images` makes the mounted folder the project's image root, so image paths
saved on another computer don't matter. The results database lands in
`/data/results/` next to the project. Export it, e.g.:

```sh
docker run --rm -v "$PWD":/data joda001/evanalyzer \
    cli results export csv --db /data/results/<run>/<run>.evadb --out /data/run1.csv
```

`cli --help` lists every command. The container runs as uid 65532: the
mounted folder must be writable for it, or pass
`--user "$(id -u):$(id -g)"`. Limit resources with Docker's `--cpus` /
`--memory`, or EVAnalyzer's `--max-threads` / `--max-ram` (MB).

## Server

```sh
docker run -d --name evanalyzer -p 7400:7400 \
    -v "$PWD":/data -v evanalyzer-state:/var/lib/evanalyzer \
    joda001/evanalyzer server --config /opt/evanalyzer/server.toml
docker logs evanalyzer   # the admin password and the TLS fingerprint
```

Connect from the desktop app or the command line with
`--remote wss://<host>:7400 --user admin --remote-fingerprint <sha256>`.
The account may use `/data` and its home. The `evanalyzer-state` volume
keeps the TLS certificate, so its fingerprint stays the same after a
restart.

Without a password in the config the server makes up a new one at every
start. For a fixed password (`evanalyzer hash-password`), more users and
other settings mount a config of your own and pass it with `--config` -
every setting is described in
[docs/server.toml](https://github.com/evanalyzer/evanalyzer/blob/main/docs/server.toml)
and [docs/server.md](https://github.com/evanalyzer/evanalyzer/blob/main/docs/server.md).

## Compute cluster (Apptainer / Singularity)

Clusters usually run Apptainer instead of Docker. It runs the image as is,
as your own user, and on any Linux - the plain Linux download needs glibc
2.39, which older cluster systems lack:

```sh
apptainer pull evanalyzer.sif docker://joda001/evanalyzer:latest
apptainer run --bind /scratch/me/study:/data evanalyzer.sif \
    --max-threads "$SLURM_CPUS_PER_TASK" --max-ram 60000 \
    cli analyze --project /data/my.evaproj --images /data/images
```

In a Slurm job script `--max-threads` and `--max-ram` (MB) should match the
job's `--cpus-per-task` and `--mem`, so EVAnalyzer plans with what the job
was given, not the whole node.

## Build the image yourself

The image is made from the release's command-line packages
([releases](https://github.com/evanalyzer/evanalyzer/releases)) on a
minimal base image (distroless, no shell) - nothing is compiled. In a
checkout of the repository, unpack them into one folder per architecture:

```sh
mkdir -p pkg/amd64 pkg/arm64
tar -xzf evanalyzer-cli-linux-x86_64.tar.gz -C pkg/amd64
tar -xzf evanalyzer-cli-linux-arm64.tar.gz  -C pkg/arm64

# both architectures (needs a buildx builder, e.g. `docker buildx create --use`)
docker buildx build -f docker/Dockerfile --build-context package=pkg \
    --platform linux/amd64,linux/arm64 --build-arg VERSION=1.2.0 -t evanalyzer .

# only amd64, loaded into the local image store
docker buildx build -f docker/Dockerfile --build-context package=pkg \
    --platform linux/amd64 --load -t evanalyzer .
```

Unreleased code:
`docker/Dockerfile.source` compiles the working copy (slow, and without the
bundled templates).

## License

EVAnalyzer is licensed under the
[PolyForm Noncommercial License 1.0.0](https://github.com/evanalyzer/evanalyzer/blob/main/LICENSE);
the licenses of the bundled third-party software are in
`/opt/evanalyzer/THIRD-PARTY-NOTICES.txt`.
