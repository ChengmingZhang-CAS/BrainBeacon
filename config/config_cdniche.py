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
MAX_LENGTH = 4096
AUX_TOKEN = 20

DATA_PATH = '/cpfs01/projects-HDD/cfff-282dafecea22_HDD/public/BrainST'
Gene_dict_path = "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/model_h5ad_1211.h5ad"
GENE_DICT_PATH = "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/model_h5ad_1211.h5ad"
GENE_LOOKUP_DIR = "/raid/zhangchengming/BrainBeacon-master/prior_knowledge/gene_lookup"
PRETRAIN_DIR = "/raid/zhangchengming/BrainBeacon-master/pretrained"