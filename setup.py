from setuptools import setup, find_packages

setup(
    name="defectlens",
    version="0.1.0",
    author="Habimana Moise",
    author_email="habimanamoise77@gmail.com",
    description="Industrial Visual Inspection and Anomaly Detection",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/moise744/defectlens-ai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "app": ["streamlit>=1.28.0"],
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
