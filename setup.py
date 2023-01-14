import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py_research",
    version="0.0.1",
    author="Nadina (Zweifel) Oates",
    author_email="nadinaoates@gmail.com",
    description="Research modules with functions for data import, plotting, signal processing, etc.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/trashpirate/py-research",
    project_urls={"Bug Tracker": "https://github.com/trashpirate/py-research/issues"},
    license="MIT",
    packages=["py_research"],
    install_requires=["numpy", "scipy", "scikit-learn", "matplotlib", "seaborn", "cycler"],
)
