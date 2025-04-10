# Triton Kernel Development Kit (TKDK)

<img src="logos/TKDK-logo.png" alt="TKDK" width="20%" height="auto">

The **Triton Kernel Development Kit (TKDK)** is a suite of tools
designed to streamline and enhance the development workflow for
Triton Kernel developers. Whether you're optimizing cache usage,
monitoring kernel performance, or distributing your builds
securely, TKDK has you covered.

## Features

### Triton Cache Manager

Organize, index, and monitor your Triton kernel caches. This tool
provides detailed reports on cache usage, offering data-driven
insights into compilation performance and cache effectiveness.

### CargoHold

Package Triton kernel caches into **OCI-compliant container images**.
Secure your caches with cryptographic signing, enabling safe and
efficient cache distribution and reuse across environments and teams.

## Getting Started

1. Clone this repository:

 ```bash
 git clone https://github.com/redhat-et/tkdk.git
 cd tkdk
 ```

1. Follow setup instructions for each tool in its respective directory.

## Project Structure

```bash
tkdk/
├── cache-manager/         # Triton Cache Manager
├── cargohold/             # OCI packaging and signing tool
└── README.md              # You're here!
```

## Security & Distribution

CargoHold ensures that your cache packages are:

- Packaged using OCI standards
- Signed cryptographically for tamper-proof integrity
- Easily distributable across environments and pipelines

## Use Cases

- Improve Triton kernel cache management
- Package and share caches across machines or Kubernetes environments.

## Contributing

We welcome contributions! If you find bugs, have feature
suggestions, or want to contribute code, please open an
issue or submit a pull request.

## License

Apache License Version 2.0. See [LICENSE](./LICENSE) for details.
