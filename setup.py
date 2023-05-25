from setuptools import setup, find_packages

PACKAGE_NAME = "stable-diffusion-service"
VERSION = "0.0.1"
DESCRIPTION = "A easy-to-use service for stable diffusion."

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,

    author="zangkai-lab",
    author_email="erickant505@gmail.com",
    long_description_content_type="text/markdown",
    keywords="stable diffusion",

    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    # 暴露为命令行工具
    entry_points={"console_scripts": ["services = services.launch:main"]},
    install_requires=[
        "click",
        "uvicorn",
        "fastapi",
        "pydantic",
        "requests",
        "python-dotenv",
        "numpy",
        "torch",
        "Pillow",
        "aiohttp",
    ],
)
