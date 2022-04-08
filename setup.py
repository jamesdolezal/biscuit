import biscuit
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="biscuit",
    version=biscuit.__version__,
    author="James Dolezal",
    author_email="james.dolezal@uchospitals.edu",
    description="Bayesian Inference of Slide-level Confidence via Uncertainty Index Thresholding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesdolezal/biscuit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'slideflow>=1.1rc0'
        'pandas'
        'click'
        'numpy'
        'tensorflow>=2.5'
        'seaborn'
        'scikit-misc'
        'scipy'
        'tqdm'
        'sklearn'
    ],
)
