from setuptools import setup, find_packages

setup(
    name="tinyserve",
    version="0.1.0",
    description="Lightweight LLM Inference Server with C++ Sampler",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "fastapi",
        "uvicorn",
        "sse-starlette",
        "psutil",
    ],
    package_data={
        "core": ["*.dll", "*.so"],
    },
)