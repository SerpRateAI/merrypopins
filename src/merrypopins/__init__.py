"""
merrypopins
========

Tools for:
 - loading indentation datasets (`load_datasets`)
 - preprocessing indentation data (`preprocess`)
 - locating pop-ins (`locate`)
 - exploring statistical models (`statistics`)
 - constructing enriched datasets (`make_dataset`)
"""

__version__ = "0.1.0"

# expose sub-modules at the package level
from . import load_datasets, preprocess, locate, statistics, make_dataset

# define what "from merrypopins import *" pulls in
__all__ = [
    "load_datasets",
    "preprocess",
    "locate",
    "statistics",
    "make_dataset",
]
