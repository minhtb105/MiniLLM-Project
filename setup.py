from setuptools import setup, find_packages


setup(
    name="minillm_project",
    version="0.0.1",
    packages=find_packages(where="src"), 
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=[
        "nltk==3.9.1",
        "numpy==2.3.3",
        "matplotlib==3.10.6",
        "pandas==2.3.2",
        "regex==2025.9.18",
        "requests==2.32.5",
        "ruff==0.13.1",
        "scikit-learn==1.7.2",
        "scipy==1.16.2",
        "setuptools==80.9.0",
        "tokenizers==0.22.1",
        "tqdm==4.67.1",
        "underthesea==8.2.0",
        "python-dotenv==1.1.1",
        "httpx==0.28.1",
        "httpcore==1.0.9"
    ],
)
