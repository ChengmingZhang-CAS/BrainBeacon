<p align="center"><img src="./docs/_static/logo_long.png" alt="BrainBeacon" width="100%"/></p>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/ChengmingZhang-CAS/BrainBeacon/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/brainbeacon

*🧠 A cross-species foundation model for single-cell–resolved brain spatial transcriptomics*


## Overview

Understanding the brain’s cellular architecture across species is fundamental to neuroscience, yet spatial transcriptomics data remain fragmented across organisms and technologies. BrainBeacon addresses this challenge through a brain-specific foundation model pretrained on BrainST-144M, a large-scale spatial brain atlas comprising 144,263,383 quality-controlled spatial observations from human, macaque, marmoset, and mouse brain datasets across MERFISH, Xenium, STARmap, Slide-seqV2, and Stereo-seq.

BrainBeacon uses a dual-stage Transformer architecture that first models intra-cellular transcriptional dependencies and then captures inter-cellular spatial context within tissue sections. Its tokenization combines ranked gene expression, gene-wise spatial deviation, local cell density, and molecular and cross-species priors. The resulting representations support zero-shot clustering and reference-guided annotation, cell-type label transfer within and across species, and representation-based virtual perturbation through gene knockout, gene overexpression, and regulon-level knockout.

<p align="center">
  <img src="./docs/_static/overview.png" width="100%" />
</p>

**Overview of the BrainBeacon pretraining corpus, dual-stage architecture, and downstream applications.**  
BrainBeacon integrates large-scale cross-species spatial brain atlases with structured tokenization of spatial transcriptomics data. The intra-cell Transformer models gene-level relationships within cells, and the inter-cell Transformer models spatial dependencies among cells within tissue sections. Downstream applications include clustering, reference-guided annotation, cross-species label transfer, and representation-based virtual perturbation.


## Installation

BrainBeacon is currently available as a development version from GitHub.

You need to have **Python 3.10 or newer** installed on your system. We recommend installing BrainBeacon in a clean virtual environment to avoid dependency conflicts.

Install the latest development version:

```bash
pip install git+https://github.com/ChengmingZhang-CAS/BrainBeacon.git@main
```

(Optional) For local development:

```bash
git clone https://github.com/ChengmingZhang-CAS/BrainBeacon.git
cd BrainBeacon
pip install -e .
```


## Documentation

Full documentation is available at:
[https://brainbeacon.readthedocs.io](https://brainbeacon.readthedocs.io)

The documentation includes installation instructions, API references, and tutorials for cell embedding and downstream analysis. Tutorials for the revised label-transfer and virtual-perturbation workflows are being updated.


## Data and Pretrained Weights

### Data overview

A summary of the spatial transcriptomics datasets used for pretraining and evaluation is provided in Supplementary Table S1, including species, platforms, and cell counts.

Processed example data and supporting resources will be deposited on Zenodo. The permanent record link will be added here after the deposition is finalized.

### Pretrained weights

Pretrained BrainBeacon checkpoints will be distributed through Zenodo. The permanent record link will be added here after the deposition is finalized.

The release contains the following checkpoints:

- `stage1_fix_step_800000.pt`
- `stage2_ep280_200.pt`

Download both files and place them directly in the `pretrained/` directory:

```text
BrainBeacon/
├── brainbeacon/
├── pretrained/
│   ├── stage1_fix_step_800000.pt
│   └── stage2_ep280_200.pt
└── ...
```

The checkpoint files are not included in the GitHub repository because of their file sizes. The gene dictionary required for tokenization is provided at `prior_knowledge/gene_dict.h5ad`.


## Citation

If you use BrainBeacon in your work, please cite:

```txt
Zhang, C., Yang, Y., et al. BrainBeacon: A cross-species foundation model for single-cell resolved brain spatial transcriptomics.
bioRxiv (2025). https://doi.org/10.1101/2025.07.08.663729        
        
```


## Contact

For questions and help requests, please open an issue on GitHub or contact the corresponding author listed in the manuscript.

[issue tracker]: https://github.com/ChengmingZhang-CAS/BrainBeacon/issues
[tests]: https://github.com/ChengmingZhang-CAS/BrainBeacon/actions/workflows/test.yaml
[documentation]: https://BrainBeacon.readthedocs.io
[changelog]: https://BrainBeacon.readthedocs.io/en/latest/changelog.html
[api documentation]: https://BrainBeacon.readthedocs.io/en/latest/api.html
