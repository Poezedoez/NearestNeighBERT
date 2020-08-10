from setuptools import setup, find_packages

print(find_packages())

setup(name='NearestNeighBERT',
      version='0.0.2',
      description='Nearest Neighbor classifier using contextual embeddings from BERT',
      author='Ragger Jonkers',
      author_email='ragger.jonkers@student.uva.nl',
      packages=find_packages(exclude=["data/", "configs/*", "scibert_scivocab_cased/", "scibert_scivocab_uncased/", "jobs/"]),
      license='LICENSE.txt',
    )

    #   