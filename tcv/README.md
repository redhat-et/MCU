# Triton Cache Vault

<img src="logo/tcv.png" alt="tcv" width="20%" height="auto">

A Triton kernel cache container packaging utility inspired by
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
$ ./_output/bin/linux_amd64/tcv -h
A GPU Kernel runtime container image management utility

Usage:
  tcv [flags]

Flags:
  -c, --create             Create OCI image
  -d, --dir string         Triton Cache Directory
  -e, --extract            Extract a Triton cache from an OCI image
  -h, --help               help for tcv
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
./_output/bin/linux_amd64/tcv -e -i quay.io/tkm/vector-add-cache:rocm
Img fetched successfully!!!!!!!!
Img Digest: sha256:b6d7703261642df0bf95175a64a01548eb4baf265c5755c30ede0fea03cd5d97
Img Size: 525
bash-4.4#
```

This will extract the cache directory from the `quay.io/tkm/vector-add-cache:rocm`
container image and copy it to  `~/.triton/cache/`.

To Create an OCI image for a Triton Cache using docker run the following:

```bash
./_output/bin/linux_amd64/tcv -c -i quay.io/tkm/vector-add-cache:rocm -d example/vector-add-cache-rocm
INFO[2025-03-28 06:44:28] baremetalFlag false
INFO[2025-03-28 06:44:28] Using docker to build the image
INFO[2025-03-28 06:44:28] Dockerfile generated successfully at /home/mtahhan/tcv/Dockerfile
{"stream":"Step 1/6 : FROM scratch"}
{"stream":"\n"}
{"stream":" ---\u003e \n"}
{"stream":"Step 2/6 : LABEL org.opencontainers.image.title=vector-add-cache"}
{"stream":"\n"}
{"stream":" ---\u003e Running in 893fef022ec3\n"}
{"stream":" ---\u003e 84dfa1409901\n"}
{"stream":"Step 3/6 : COPY \"example/vector-add-cache/\" ./io.triton.cache/"}
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
{"stream":"Successfully tagged quay.io/tkm/vector-add-cache:latest\n"}
INFO[2025-03-28 06:44:42] Temporary directories successfully deleted.
INFO[2025-03-28 06:44:42] Docker image built successfully
INFO[2025-03-28 06:44:42] OCI image created successfully.
```

To see the new image:

```bash
 docker images
REPOSITORY                                                                                TAG                   IMAGE ID       CREATED          SIZE
quay.io/tkm/vector-add-cache                                                       latest                32572653bbbd   5 minutes ago    0B
```

To inspect the docker image with Skopeo

```bash
skopeo inspect docker://quay.io/tkm/vector-add-cache
{
    "Name": "quay.io/tkm/vector-add-cache",
    "Digest": "sha256:9e64f49656f89d5d68a739dad2d8b333ab258a77f70efe2f8961b8752f0ef0fd",
    "RepoTags": [],
    "Created": "2025-03-28T10:44:42.611261047Z",
    "DockerVersion": "27.3.1",
    "Labels": {
        "cache.triton.image/entry-count": "1",
        "cache.triton.image/metadata": "[{\"hash\":\"edbea4c0734897ca19ec52852fcc847e552b3b1cfff92cc3deff0b695cdd636f\",\"backend\":\"cuda\",\"arch\":\"75\",\"warp_size\":32,\"dummy_key\":\"f057a3304cf191347dfefc46ce6def3d0120a5abeb7373154bb72a8256c80413\"}]",
        "cache.triton.image/variant": "multi",
        "org.opencontainers.image.title": "vector-add-cache"
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
./_output/bin/linux_amd64/tcv -c -i quay.io/tkm/vector-add-cache:rocm -d example/vector-add-cache-rocm
INFO[2025-05-28 11:08:15] baremetalFlag false
INFO[2025-05-28 11:08:15] Using buildah to build the image
INFO[2025-05-28 11:08:15] Wrote manifest to /tmp/buildah-manifest-dir-392890374/manifest.json
INFO[2025-05-28 11:08:15] Image built! b81c4953849f4b4d97cce319eee59d267154ebd10677347d1900cebcbd0a4e0a
INFO[2025-05-28 11:08:15] Temporary directories successfully deleted.
INFO[2025-05-28 11:08:15] OCI image created successfully.
```

To inspect the buildah image with Skopeo

```bash
skopeo inspect containers-storage:quay.io/tkm/vector-add-cache:rocm
{
    "Name": "quay.io/tkm/vector-add-cache",
    "Digest": "sha256:959dd97f5934a9e2e1bf2cf41304a2b55fa8eb89bb47c69810d04975e07e06f7",
    "RepoTags": [],
    "Created": "2025-05-28T15:08:15.773233031Z",
    "DockerVersion": "",
    "Labels": {
        "cache.triton.image/cache-size-bytes": "80415",
        "cache.triton.image/entry-count": "1",
        "cache.triton.image/summary": "{\"variant\":\"multi\",\"entry_count\":1,\"targets\":[{\"backend\":\"hip\",\"arch\":\"gfx90a\",\"warp_size\":64}]}",
        "cache.triton.image/variant": "multi"
    },
    "Architecture": "amd64",
    "Os": "linux",
    "Layers": [
        "sha256:872c6b1635496d986ab539cbbb5368f47a0bf7bede66495876553ee326986dd3"
    ],
    "LayersData": [
        {
            "MIMEType": "application/vnd.oci.image.layer.v1.tar",
            "Digest": "sha256:872c6b1635496d986ab539cbbb5368f47a0bf7bede66495876553ee326986dd3",
            "Size": 93184,
            "Annotations": null
        }
    ],
    "Env": null
}
```

To inspect the image labels specifically run:

```bash
skopeo inspect containers-storage:quay.io/tkm/vector-add-cache:rocm \
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
