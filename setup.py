import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bitsota",
    version="0.0.1",
    description="A distributed AutoML package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AlveusLabs/SN94-BitSota/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "bitsota-research-agent=neurons.research_agent_miner:main",
            "bitsota-research-validator=validator.research_validator_runner:main",
            "bitsota-claim-rewards=gui.merkle_claim_cli:main",
        ],
    },
)
