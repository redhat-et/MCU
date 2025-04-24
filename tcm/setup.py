from setuptools import setup, find_packages

setup(
    name="triton-cache-manager",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["typer[all]", "structlog", "pydantic>=2", "pydantic-settings>=2"],
    entry_points={
        "console_scripts": ["triton-cache=triton_cache_manager.cli.main:run"]
    },
    python_requires=">=3.9",
)
