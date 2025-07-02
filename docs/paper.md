---
title: 'merrypopins: A Python package for nanoindentation data science'
tags:
  - Python
  - geology
  - nanoindentation
  - geophysics
authors:
  - name: Cahit Acar
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Anna Mercelissen
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Hugo W. van Schrojenstein
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: John M. Aiken
  - affiliation: "1, 2, 3"
affiliations:
 - name: Utrecht University, The Netherlands
   index: 1
 - name: Expert Analytics, Norway
   index: 2
 - name: University of Oslo, Njord Centre, Norway
   index: 3
date: 2 July 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

merrypopins is a Python library to streamline the workflow of nano‑indentation 
experiment data processing, automated pop-in detection and analysis. merrypopins 
is an open-source Python library that implements four unsupervised
pop-in detectors: Savitzky-Golay filtering, Fourier-domain differ-
entiation, Isolation Forest anomaly detection, and a convolutional
autoencoder.
Understanding the start of plasticity in materials at the microscale
is crucial for various applications, including engineered materials
and earthquake mechanics. Investigations into nano-indentation
reveal sudden "pop-in" events that cause significant spikes in in-
dentation depth along load-depth curves under almost constant
force. Manually recognizing these characteristics is labor-intensive
and subjective, emphasizing the importance of automated, repro-
ducible detection approaches. This thesis outlines 

We utilize merrypopins in 30 slow-loading experiments on glau-
cophane grains in a thin slice of blueschist (60 s ramp, 200 mN peak
load) utilizing a 6 µm cono-spherical tip on a Hysitron TI 990 Tri-
boIndenter. High agreement across experiments, few false positives,
and tight, physically reasonable clusters of pop-in depths (median
≈650 nm) are the results of analytical derivative-based techniques
(Savitzky–Golay and Fourier). On the other hand, the autoencoder
and isolation forest methods generate many shallow-depth detec-
tions that have a weak correlation with actual material instabilities.
These findings show that conventional signal-processing methods
offer reliable, comprehensible baselines for pop-in detection with
slight adjustment of parameters.
This study uses an anomaly-detection task on univariate time
series to demonstrate the reliability of automated approaches in
capturing pop-in occurrences and highlighting trade-offs between
sensitivity, transparency, and complexity. The merrypopins library
provides an integrated pipeline for loading, preparing, locating,
and visualizing nano-indentation data. It enables researchers to use,
expand, and improve pop-in analysis, opening the way for version 2
developments, including semi-supervised lear

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
