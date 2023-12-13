import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()
    
setuptools.setup(
        name="Biomedical Event finder",
        version="0.0.1",
        author="JC. Rangel",
        author_email="juliocesar.rangelreyes@riken.jp",
        description="Biomedical events searcher",
        long_description=long_description,
        long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        include=['faiss_utils', 'bef','DeepEventMine'])
    )
