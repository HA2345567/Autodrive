import os
from setuptools import setup, find_packages

# Read requirements.txt
def read_requirements():
    reqs_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(reqs_path):
        with open(reqs_path, 'r', encoding='utf-8') as f:
            return [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('#')
            ]
    return []

# Read long description from README.md if it exists
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="autodrive",
    version="0.1.0",
    description="An autonomous driving system featuring steering angle prediction and YOLO v11 object detection",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="AutoDrive Developers",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
