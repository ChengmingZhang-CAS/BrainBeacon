# Pretrained Weights for BrainBeacon

The pretrained BrainBeacon checkpoints are distributed as
`brainbeacon_pretrained_checkpoints.zip` through Zenodo. The permanent Zenodo
record and DOI will be added here after the deposition is finalized.

## Checkpoints

The archive contains:

- `stage1_fix_step_800000.pt`: pretrained Stage 1 molecular encoder checkpoint.
- `stage2_ep280_200.pt`: pretrained Stage 2 inter-cell spatial model checkpoint.

Keep these filenames unchanged because the repository configuration, examples,
and tutorials use the exact names shown above.

## Installation

Download and extract `brainbeacon_pretrained_checkpoints.zip`, then place both
checkpoint files directly in the `pretrained/` directory:

```text
BrainBeacon/
|-- brainbeacon/
|-- pretrained/
|   |-- README.md
|   |-- stage1_fix_step_800000.pt
|   `-- stage2_ep280_200.pt
|-- prior_knowledge/
|   `-- gene_dict.h5ad
`-- ...
```

The checkpoint files are not included in the GitHub repository because of their
file sizes. The gene dictionary required for tokenization is distributed with
the repository at `prior_knowledge/gene_dict.h5ad`.

## License

The pretrained model checkpoints are released under the Creative Commons
Attribution 4.0 International (CC BY 4.0) license. The BrainBeacon source code
is distributed separately under the MIT License; see the repository-level
[`LICENSE`](../LICENSE) file.

