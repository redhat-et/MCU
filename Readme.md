# Model Cache Utils (MCU)

<img src="logo/mcu.png" alt="MCU" width="20%" height="auto">

The **Model Cache Utils (MCU)** (formerly Triton Kernel Development
Kit (TKDK)) is a suite of tools designed to streamline and enhance
the development workflow for Model Kernel developers. Whether you're
optimizing cache usage, monitoring kernel performance, or distributing
your builds securely, MCU has you covered. MCU supports Triton and vLLM.

## Features

### Model Cache Manager (MCM)

Organize, index, and monitor your Model kernel caches. This tool
provides detailed reports on cache usage, offering data-driven
insights into compilation performance and cache effectiveness.
For more information please see the MCM [readme](./mcm/README.md).

### Model Cache Vault (MCV) - MOVED

**MCV has been moved to its own repository**:
[https://github.com/redhat-et/GKM](https://github.com/redhat-et/GKM)

MCV (Model Cache Vault) packages Model/GPU kernel caches into
**OCI-compliant container images** with cryptographic signing for
secure cache distribution. Please refer to the new repository for
the latest features and documentation.

### Triton Util

Write cleaner, more intuitive Triton code with high-level abstractions
and utilities for loading, storing, and debugging GPU memory.

**Triton-util was developed by [Umer Adil](mailto:umer.hayat.adil@gmail.com)**
and generously contributed to MCU.

For more information please see the Triton Util [readme](./triton_util/README.md).

## Getting Started

1. Clone this repository:

    ```bash
    git clone https://github.com/redhat-et/MCU.git
    cd MCU
    ```

1. Follow setup instructions for each tool in its respective directory.

## Project Structure

```bash
MCU/
├── mcm/           # Model Cache Manager
├── triton_util/   # Triton Utilities
└── README.md      # You're here!
```

## Use Cases

- Improve Triton/vLLM kernel cache management with MCM
- For packaging and sharing caches, see [GKM (formerly MCV)](https://github.com/redhat-et/GKM)

## Contributing

We welcome contributions! If you find bugs, have feature
suggestions, or want to contribute code, please open an
issue or submit a pull request.

## License

Apache License Version 2.0. See [LICENSE](./LICENSE) for details.
