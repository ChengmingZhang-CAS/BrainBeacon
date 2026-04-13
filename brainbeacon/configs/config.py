specie_dict = {
    'human': 3,
    'mouse': 4,
    'macaque': 5,
    'marmoset': 6
}

technology_dict = {
    "merfish": 7,
    "MERFISH": 7,
    "Xenium": 8,
    "xenium": 8,
    "XENIUM": 8,
    "STARmap": 9,
    "starmap": 9,
    "STARMAP": 9,
    "SlideSeqV2": 10,
    "slideseqv2": 10,
    "SlideseqV2": 10,
    "stereo": 11,
    "snrna": 12,
    "snRNA": 12,
}
cell_density_bin_dict = {
    "cell_density_bin_0": 15,
    "cell_density_bin_1": 16,
    "cell_density_bin_2": 17,
    "cell_density_bin_3": 18,
    "cell_density_bin_4": 19,
}
MAX_LENGTH = 1000
AUX_TOKEN = 20

DEFAULT_PATHS = {
    "DATA_PATH": "/cpfs01/projects-HDD/cfff-282dafecea22_HDD/public/BrainST",
    "BASE_DIR": "/raid/zhangchengming/BrainBeacon-master",
    "PRETRAIN_DIR": "/raid/zhangchengming/BrainBeacon-master/pretrained",
    "PRIOR_DIR": "/raid/zhangchengming/BrainBeacon-master/prior_knowledge",
    # "GENE_DICT_PATH": "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/model_h5ad_1211.h5ad",
    "GENE_DICT_PATH": "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/gene_dict.h5ad",
    "GENE_LOOKUP_DIR": "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/gene_lookup",
    "ESM_EMBED_PATH": "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/esm2_embeddings_d5120.pt",
}

# DEFAULT_PATHS = {
#     "BASE_DIR": "/need_file_path/BrainBeacon-master",
#     "PRETRAIN_DIR": "/need_file_path/BrainBeacon-master/pretrained",
#     "PRIOR_DIR": "/need_file_path/BrainBeacon-master/prior_knowledge",
#     "GENE_DICT_PATH": "/need_file_path/prior_knowledge/model_h5ad_1211.h5ad",
#     "GENE_LOOKUP_DIR": "/need_file_path/prior_knowledge/gene_lookup",
#     "ESM_EMBED_PATH": "/need_file_path/prior_knowledge/esm2_embeddings_d5120.pt",
# }

def resolve_path(key: str, path_dict: dict | None = None) -> str:
    """
    Pick a path with priority:
      1) path_dict[key] if provided and non-empty
      2) DEFAULT_PATHS[key]
    """
    if path_dict is not None:
        v = path_dict.get(key, None)
        if v is not None and str(v).strip() != "":
            return str(v)
    return DEFAULT_PATHS[key]
