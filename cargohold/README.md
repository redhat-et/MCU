# CargoHold

<img src="logo/cargo-hold.png" alt="cargohold" width="20%" height="auto">

A GPU Kernel runtime container packaging utility inspired by
[WASM](https://github.com/solo-io/wasm/blob/master/spec/README.md).

## Build

```bash
sudo dnf install gpgme-devel
sudo dnf install btrfs-progs-devel
```

```bash
go build
```

## Usage

```bash
$ ./_output/bin/linux_amd64/cargohold -h
A GPU Kernel runtime container image management utility

Usage:
  cargohold [flags]

Flags:
  -c, --create             Create OCI image
  -d, --dir string         Triton Cache Directory
  -e, --extract            Extract a Triton cache from an OCI image
  -h, --help               help for cargohold
  -i, --image string       OCI image name
  -l, --log-level string   Set the logging verbosity level (debug, info, warning or error)
```

> NOTE: The create option is a work in progress. For now
to create an OCI image containing a Triton cache directory
please follow the instructions in
[spec-compat.md](./spec-compat.md).

## Dependencies

- [buildah dependencies](https://github.com/containers/buildah/blob/main/install.md#building-from-scratch)

## Triton Cache Image Container Specification

The Triton Cache Image specification defines how to bundle Triton Caches
as container images. A compatible Triton Cache image consists of cache
directory for a Triton Kernel.

There are two variants of the specification:

- [spec.md](./spec.md)
- [spec-compat.md](./spec-compat.md)

## Example

To extract the Triton Cache for the
[01-vector-add.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)
tutorial from [Triton](https://github.com/triton-lang/triton), run the following:

```bash
./_output/bin/linux_amd64/cargohold -e -i quay.io/mtahhan/triton-cache:01-vector-add-latest
Img fetched successfully!!!!!!!!
Img Digest: sha256:b6d7703261642df0bf95175a64a01548eb4baf265c5755c30ede0fea03cd5d97
Img Size: 525
bash-4.4#
```

This will extract the cache directory from the `quay.io/mtahhan/triton-cache:01-vector-add-latest`
container image and copy it to  `~/.triton/cache/`.

To Create an OCI image for a Triton Cache using docker run the following:

```bash
./_output/bin/linux_amd64/cargohold -c -i quay.io/mtahhan/01-vector-add-cache -d example/01-vector-add-cache
INFO[2025-03-28 06:44:28] baremetalFlag false
INFO[2025-03-28 06:44:28] Using docker to build the image
INFO[2025-03-28 06:44:28] Dockerfile generated successfully at /home/mtahhan/cargohold/Dockerfile
{"stream":"Step 1/6 : FROM scratch"}
{"stream":"\n"}
{"stream":" ---\u003e \n"}
{"stream":"Step 2/6 : LABEL org.opencontainers.image.title=01-vector-add-cache"}
{"stream":"\n"}
{"stream":" ---\u003e Running in 893fef022ec3\n"}
{"stream":" ---\u003e 84dfa1409901\n"}
{"stream":"Step 3/6 : COPY \"example/01-vector-add-cache/\" ./io.triton.cache/"}
{"stream":"\n"}
{"stream":" ---\u003e a009d449e513\n"}
{"stream":"Step 4/6 : LABEL cache.triton.image/entry-count=1"}
{"stream":"\n"}
{"stream":" ---\u003e Running in eee936b013ac\n"}
{"stream":" ---\u003e 68ed9b0860aa\n"}
{"stream":"Step 5/6 : LABEL cache.triton.image/metadata=[{\"hash\":\"edbea4c0734897ca19ec52852fcc847e552b3b1cfff92cc3deff0b695cdd636f\",\"backend\":\"cuda\",\"arch\":\"75\",\"warp_size\":32,\"dummy_key\":\"f057a3304cf191347dfefc46ce6def3d0120a5abeb7373154bb72a8256c80413\"}]"}
{"stream":"\n"}
{"stream":" ---\u003e Running in ce100bca19f4\n"}
{"stream":" ---\u003e c4fb6e41ee67\n"}
{"stream":"Step 6/6 : LABEL cache.triton.image/variant=multi"}
{"stream":"\n"}
{"stream":" ---\u003e Running in 25572b1371d2\n"}
{"stream":" ---\u003e cdb197dc95ab\n"}
{"aux":{"ID":"sha256:cdb197dc95ab2a4c1e6a40583fe22c26d65660c0681e3559ab70d1b9c1b2f79d"}}
{"stream":"Successfully built cdb197dc95ab\n"}
{"stream":"Successfully tagged quay.io/mtahhan/01-vector-add-cache:latest\n"}
INFO[2025-03-28 06:44:42] Temporary directories successfully deleted.
INFO[2025-03-28 06:44:42] Docker image built successfully
INFO[2025-03-28 06:44:42] OCI image created successfully.
```

To see the new image:

```bash
 docker images
REPOSITORY                                                                                TAG                   IMAGE ID       CREATED          SIZE
quay.io/mtahhan/01-vector-add-cache                                                       latest                32572653bbbd   5 minutes ago    0B
```

To inspect the docker image with Skopeo

```bash
{
    "Name": "quay.io/mtahhan/01-vector-add-cache",
    "Digest": "sha256:9e64f49656f89d5d68a739dad2d8b333ab258a77f70efe2f8961b8752f0ef0fd",
    "RepoTags": [],
    "Created": "2025-03-28T10:44:42.611261047Z",
    "DockerVersion": "27.3.1",
    "Labels": {
        "cache.triton.image/entry-count": "1",
        "cache.triton.image/metadata": "[{\"hash\":\"edbea4c0734897ca19ec52852fcc847e552b3b1cfff92cc3deff0b695cdd636f\",\"backend\":\"cuda\",\"arch\":\"75\",\"warp_size\":32,\"dummy_key\":\"f057a3304cf191347dfefc46ce6def3d0120a5abeb7373154bb72a8256c80413\"}]",
        "cache.triton.image/variant": "multi",
        "org.opencontainers.image.title": "01-vector-add-cache"
    },
    "Architecture": "amd64",
    "Os": "linux",
    "Layers": [
        "sha256:2bb77dbac0435c7f0c5e1524dba0954c99e250bac7fd3fc6b9ef57e742e43278"
    ],
    "LayersData": [
        {
            "MIMEType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
            "Digest": "sha256:2bb77dbac0435c7f0c5e1524dba0954c99e250bac7fd3fc6b9ef57e742e43278",
            "Size": 82432,
            "Annotations": null
        }
    ],
    "Env": [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ]
}
```

> **Note**: If `buildah` is installed it will be favoured to build the image.
The build output is shown below.

```bash
 ./_output/bin/linux_amd64/cargohold -c -i quay.io/mtahhan/01-vector-add-cache -d example/01-vector-add-cache
INFO[2025-03-28 10:20:11] baremetalFlag false
INFO[2025-03-28 10:20:11] Using buildah to build the image
INFO[2025-03-28 10:20:11] Image built! 9def3c99415d1716e94eb6a2b010b2010c80de81c78608612e4bec1c21d27e62
INFO[2025-03-28 10:20:11] Temporary directories successfully deleted.
INFO[2025-03-28 10:20:11] OCI image created successfully.
```

To inspect the buildah image with Skopeo

```bash
skopeo inspect containers-storage:quay.io/mtahhan/01-vector-add-cache:latest
{
    "Name": "quay.io/mtahhan/01-vector-add-cache",
    "Digest": "sha256:2a5ce196b565d00b3e7afcf01bb9e6abdb52333f13b1da345201820553847817",
    "RepoTags": [],
    "Created": "2025-03-28T10:20:11.890750347Z",
    "DockerVersion": "",
    "Labels": {
        "cache.triton.image/entry-count": "1",
        "cache.triton.image/metadata": "[{\"hash\":\"edbea4c0734897ca19ec52852fcc847e552b3b1cfff92cc3deff0b695cdd636f\",\"backend\":\"cuda\",\"arch\":\"75\",\"warp_size\":32,\"dummy_key\":\"f057a3304cf191347dfefc46ce6def3d0120a5abeb7373154bb72a8256c80413\"}]",
        "cache.triton.image/variant": "multi"
    },
    "Architecture": "amd64",
    "Os": "linux",
    "Layers": [
        "sha256:710625569a9e7ad0ba7af6faa63ced5ff17cbac17a9750f166465eb48a7d5071"
    ],
    "LayersData": [
        {
            "MIMEType": "application/vnd.oci.image.layer.v1.tar",
            "Digest": "sha256:710625569a9e7ad0ba7af6faa63ced5ff17cbac17a9750f166465eb48a7d5071",
            "Size": 82944,
            "Annotations": null
        }
    ],
    "Env": null
}
```

To inspect the image labels specifically run:

```bash
skopeo inspect containers-storage:quay.io/mtahhan/01-vector-add-cache:latest \
  | jq -r '.Labels["cache.triton.image/metadata"]' \
  | jq .
[
  {
    "hash": "edbea4c0734897ca19ec52852fcc847e552b3b1cfff92cc3deff0b695cdd636f",
    "backend": "cuda",
    "arch": "75",
    "warp_size": 32,
    "dummy_key": "f057a3304cf191347dfefc46ce6def3d0120a5abeb7373154bb72a8256c80413"
  }
]
```
