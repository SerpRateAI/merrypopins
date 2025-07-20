---
title: '`merrypopins`: A Python package for nanoindentation data science'
tags:
  - Python
  - geology
  - nanoindentation
  - deformation
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
  - name: Alissa J. Kotowski
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

`merrypopins` is a Python library to streamline the workflow of nanoindentation experiment data processing, automated pop-in detection, and statistical analysis of collections of pop-in events. Nanoindentation is a technique for experimental deformation of materials with the aim of characterizing material behaviour and quantifying mechanical properties from load-displacement data [@oliverpharr1992indentation, @oliverpharr2004indentation]. Experiments performed with a spherical tip can also be used to construct stress-strain curves and by extension the determination of the yield point defining the transition from elastic to plastic deformation [@kalidindipathak2008stressstrain, @pathakkalidindi2015stressstrain]. Understanding the start of plasticity in materials at the microscale is crucial for various applications, including engineered materials and earthquake mechanics. A common feature during the loading part of nanoindentation experiments is the sudden increase of indentation depth at constant force, called "pop-in" events. Manually recognizing these characteristics is labor-intensive and subjective, emphasizing the importance of automated, reproducible detection approaches. 

# Statement of need

Despite their importance, detecting pop-ins is difficult because they appear in subtle, intermittent, and different ways within indentation curves. Historically, professional analysts have recognized pop-in occurrences manually. The researcher simply looks at either depth vs. time or stress-strain curves looking for sharp, localized changes. This approach suffers from subjectivity, labor intensity, and potential inconsistencies among multiple observers and big datasets. Modern nano-indentation machines can perform up to 12 indentations per second [@bruker] and thus new tools to automate pop-in detection are necessary to be built in order to keep up with the increase in data. `merrypopins` marks the first attempt to automate pop-in detection. 

Pop-ins are linked to dislocation in crystalline materials and are considered small-scale analogues of earthquakes [@ispanovity2022dislocation, @sato2020unique]. Like real earthquakes, they follow statistical patterns, such as power-law distributions in size and time between events. Generally, the first pop-in during an indentation experiment coincides with the start of plasticity. The size of the indenter tip and the degree of pre-existing plastic deformation have a significant impact on the stress at which the first pop-in occurs [@shim2008sizeeffect, @morris2011sizeeffect] and thus the yield hardness of the material. This effect is the result of a delayed plastic yielding when the volume stressed by the indenter tip does not contain any pre-existing dislocations for the initiation of plasticity. A smaller volume is more likely to be free of dislocations, especially when the material has a lower dislocation density, so the material will behave elastically up to higher load and stress. In contrast, larger tips sample a bigger volume, increasing the chance of hitting existing dislocations and causing the first pop-in at lower stresses. This size effect must be overcome to obtain yield hardness values applicable across scales or to other systems.

The primary stakeholders of `merrypopins` are students, researchers, and academics in the fields of material science, geology, nano-mechanics, and earthquake science. High-resolution indentation experiments are increasingly used to investigate plastic and fracture processes at the microscale. Despite the growing number of studies targeting pop-in occurrences in load-depth curves, almost all previous research relies on manual inspection or private scripts with undisclosed methods for the detection and quantification of pop-ins, creating a lack of easily accessible, reproducible event detection software. There is an urgent need for adaptable, open-source solutions that can be used "out of the box" by non-programmers and provide extensibility for power users as nanoindentation tools grow, spanning both traditional materials laboratories and emerging geophysical applications. To advance the next generation of automated pop-in analysis, researchers can submit new detection techniques, parameter settings, or visualization modules through our public `merrypopins` GitHub repository. We, therefore, welcome feature requests, bug reports, and community-contributed enhancements.

Using a variety of detection techniques ensures that `merrypopins` can detect pop-in events in many material systems and experimetal circumstances. `merrypopins` primary uses the Savitzky-Golay filter and Fourier-domain differentiation methods for pop-in detection. Savitzky-Golay’s local polynomial smoothing maintains prominent curve characteristics while reducing high-frequency noise [@savitzky1964smoothing]. Fourier spectral methods identify abrupt discontinuities with minimal parameterization [@cooley2007fast]. Both strategies are computationally efficient, highly interpretable, and require only a few user-tunable parameters (window length, polynomial order, or frequency threshold), making them excellent for quick initial screening.

`merrypopins` also includes two other machine learning methods. Isolation Forest and convolutional autoencoders enable data-driven adaptation. Isolation Forest, an unsupervised ensemble-based statistical framework, can detect anomalies in multidimensional feature spaces without labeled instances (@liu2008isolation). This is especially useful when the pop-in magnitudes or frequencies are unknown beforehand. Convolutional autoencoders learn hierarchical feature representations directly from data, capturing subtle nonlinear patterns that classical approaches may overlook
[@malhotra2016lstm]. However, they require more resources. These four techniques balance sensitivity, interpretability, and processing cost, allowing researchers to select and combine algorithms based on dataset size, noise characteristics, and analytic goals.

The `merrypopins` library was developed using a tutorial-driven software development framework [@tutorial]. Instead of starting with predetermined architectural specs, this approach converts the scientist’s process into a live, executable lesson (often a Jupyter notebook). Developers and researchers worked iteratively, with academics creating function stubs in a scientific narrative framework and developers implementing these functions based on real-world usage cases. This strategy ensures that scientific usability drives software design. 

# Code Availability

The `merrypopins` package can be installed via:

```
pip install `merrypopins`
```

Alternatively, the package can be found on github ([https://github.com/SerpRateAI/`merrypopins`](https://github.com/SerpRateAI/`merrypopins`)).

Contributions can be made by forking the repository and making a pull request.

The streamlit app is accessible via the streamlit website ([https://`merrypopins`.streamlit.app/](https://`merrypopins`.streamlit.app/)).

# Acknowledgements

This project has received funding from the Norwegian Research Council (SerpRateAI, grant no. 334395) and is supported by EPOS-eNLarge funded by the Dutch Research Council (NWO) Roadmap for large-scale research infrastructure.

# References
