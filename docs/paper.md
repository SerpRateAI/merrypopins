---
title: '`merrypopins`: A Python package for nanoindentation data science'
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

`merrypopins` is a Python library to streamline the workflow of nano‑indentation 
experiment data processing, automated pop-in detection and analysis.
Understanding the start of plasticity in materials at the microscale
is crucial for various applications, including engineered materials
and earthquake mechanics. Investigations into nano-indentation
reveal sudden "pop-in" events that cause significant spikes in in-
dentation depth along load-depth curves under almost constant
force. Pop-ins are linked to dislocation in crystalline materials and are consid-
ered small-scale analogues of earthquakes [@ispanovity2022dislocation, @sato2020unique]. Like real earthquakes,
they follow statistical patterns, such as power-law distributions in size and
time between events. Furthermore, the size of the indenter tip affects
when a pop-in occurs. Smaller tips often lead to higher pop-in stresses be-
cause they are more likely to probe regions without dislocations. In contrast,
larger tips sample a bigger volume, increasing the chance of hitting existing
dislocations and causing pop-ins at lower stresses.  Manually recognizing these characteristics is labor-intensive
and subjective, emphasizing the importance of automated, repro-
ducible detection approaches. 

# Statement of need

Despite their importance, detecting pop-ins is difficult because they
appear in subtle, intermittent, and different ways within inden-
tation curves. Historically, professional analysts have recognized
pop-in occurrences manually, but this approach suffers from subjec-
tivity, labor intensity, and potential inconsistencies among multiple
observers and big datasets.



The primary stakeholders of `merrypopins` are students, researchers,
and academics in the fields of materials science, geology, nano-
mechanics, and earthquake science. High-resolution indentation
experiments are increasingly used to investigate plastic and frac-
ture processes at the microscale. Despite the growing number of
pop-in occurrences in load-depth curves, almost all previous re-
search relies on manual inspection or private scripts, creating a lack
of easily accessible, reproducible event detection software. There
is an urgent need for adaptable, open-source solutions that can
be used "out of the box" by non-programmers and provide exten-
sibility for power users as nanoindentation tools grow, spanning
both traditional materials laboratories and emerging geophysical
applications. To advance the next generation of automated
pop-in analysis, researchers can submit new detection techniques,
parameter settings, or visualization modules through our public
`merrypopins` GitHub repository. We, therefore, welcome feature
requests, bug reports, and community-contributed enhancements.

Using a variety of detection techniques ensures that `merrypopins`
can detect pop-in events in many material systems and experimen-
tal circumstances. The Savitzky-Golay filter and Fourier-domain
differentiation provide physics-based baselines. Savitzky-Golay’s
local polynomial smoothing maintains prominent curve character-
istics while reducing high-frequency noise [@savitzky1964smoothing]. Fourier spectral
methods identify abrupt discontinuities with minimal parameter-
ization [6]. Both strategies are computationally efficient, highly
interpretable, and require only a few user-tunable parameters (win-
dow length, polynomial order, or frequency threshold), making
them excellent for quick initial screening.

In contrast, Isolation Forest and convolutional autoencoders
enable data-driven adaptation. Isolation Forest, an unsupervised
ensemble-based statistical framework, can detect anomalies in mul-
tidimensional feature spaces without labeled instances (Liu, 2008).
This is especially useful when the pop-in magnitudes or frequen-
cies are unknown beforehand. Convolutional autoencoders learn
hierarchical feature representations directly from data, capturing
subtle nonlinear patterns that classical approaches may overlook
[@malhotra2016lstm]. However, they require more resources. These four techniques
balance sensitivity, interpretability, and processing cost, allowing
researchers to select and combine algorithms based on dataset size,
noise characteristics, and analytic goals.

The `merrypopins` library was developed using a tutorial-driven
software development framework. Instead of starting with predetermined 
architectural specs, this approach converts the scientist’s
process into a live, executable lesson (often a Jupyter notebook).
Developers and researchers worked iteratively, with academics
creating function stubs in a scientific narrative framework and de-
velopers implementing these functions based on real-world usage
cases. This strategy ensures that scientific usability drives software
design. This technique is consistent with Christopher Woods’ para-
digm2 and has successfully bridged the gap between exploratory
research code and reusable scientific software.

# Code Availability

The `merrypopins` package can be installed via:

```
pip install `merrypopins`
```

Alternatively, the package can be found on github ([https://github.com/SerpRateAI/`merrypopins`](https://github.com/SerpRateAI/`merrypopins`)).

Contributions can be made by forking the repository and making a pull request.

The streamlit app is accessible via the streamlit website ([https://`merrypopins`.streamlit.app/](https://`merrypopins`.streamlit.app/)).

<!-- 
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
[@gaia] by students and experts alike. -->

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
