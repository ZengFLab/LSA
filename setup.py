from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LSA-sc',
    version='0.1.0',
    description='Latent Space Alignment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Feng Zeng',
    author_email='zengfeng@xmu.edu.cn',
    packages=find_packages(),
    install_requires=['dill==0.3.8','scanpy','pytorch-ignite','datatable','scipy','numpy','scikit-learn','pandas','pyro-ppl', "jax[cuda12]",
                      'leidenalg','python-igraph','networkx','matplotlib','seaborn','fa2-modified','zuko','plotly'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    url='https://github.com/ZengFLab/LSA',  # 项目的 GitHub 地址

)