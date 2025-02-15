from setuptools import setup, find_packages

setup(
    name="rxivonauta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.5",
        "arxiv>=1.4.8",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "tenacity>=8.2.3",
        "asyncio>=3.4.3",
        "typing-extensions>=4.7.1",
        "tqdm>=4.65.0",
        "openai>=1.0.0"
    ],
    entry_points={
        'console_scripts': [
            'rxivonauta=src.main:main',
        ],
    },
    python_requires=">=3.8",
)
