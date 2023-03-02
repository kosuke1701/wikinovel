from glob import glob
from setuptools import setup

prompt_files = glob("src/wikinovel/prompts/*")

setup(
    name="wikinovel",
    version="0.0.0",
    install_requires=[
        "langchain"
    ],
    author="Kosuke Akimoto",
    author_email="kosuke1701@gmail.com",
    package_dir={"": "src"},
    data_files=[
        ("prompts", prompt_files),
    ],
    include_package_data=True,
)