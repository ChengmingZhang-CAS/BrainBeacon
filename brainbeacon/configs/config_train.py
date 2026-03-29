# config_train.py - Yifeng's version
platform_prob = {
    11: 0.8268,
    7: 0.1327,
    9: 0.0130,
    10: 0.0188,
    8: 0.0085
}

species_prob = {
    6: 0.2000,
    4: 0.2610,
    5: 0.5015,
    3: 0.0375
}

config_train0 = {
    'pretrained_path': None,
    'retake_training': False,
    'dim_feedforward': 768,
    'nheads': 8,
    'masking_p': 0.15,
    'nlayers': 8,
    'dropout': 0.1,
    'dim_model': 768,
    'batch_first': True,
    'n_tokens': 92076,
    'n_connect_comp': 46502,
    'n_aux': 20,
    'n_rna_type': 33,
    'batch_size': 16,
    'context_length': 1000,
    'lr': 1e-5,
    'warmup': 1000,
    'max_epoch': 100,
    'total_steps': 100000,
    'autoregressive': False,
    'pool': None,
    'supervised_task': False,
    'learnable_pe': True,
    'organ': "everything",
    'specie': True,
    'assay': True,
    'modality': False,
    'contrastive': False,
    'neighbor_enhance': True,
    'num_neighbors': 4,
    'use_esm_embedding': True,
    'ems_embedding_dim': 5120,
    'root_path': '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST',
    'esm_embedding_path': '/raid/zhangchengming/BrainBeacon-master/prior_knowledge/xesm2_embeddings_d5120.pt',
    'pretrain_ckpt': None
}

config_train = {
    'pretrained_path': None,
    'retake_training': False,
    'dim_feedforward': 1024,
    'nheads': 16,
    'masking_p': 0.15,
    'nlayers': 16,
    'dropout': 0.1,
    'dim_model': 1024,
    'batch_first': True,
    'n_tokens': 92076,
    'n_connect_comp': 46502,
    'n_aux': 20,
    'n_rna_type': 33,
    'batch_size': 16,
    'context_length': 1000,
    'lr': 1e-5,
    'warmup': 1000,
    'max_epoch': 100,
    'total_steps': 100000,
    'autoregressive': False,
    'pool': None,
    'supervised_task': False,
    'learnable_pe': True,
    'organ': "everything",
    'specie': True,
    'assay': True,
    'modality': False,
    'contrastive': False,
    'neighbor_enhance': True,
    'num_neighbors': 4,
    'use_esm_embedding': True,
    'ems_embedding_dim': 5120,
    'root_path': '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST',
    # 'esm_embedding_path': '/raid/zhangchengming/BrainBeacon-master/prior_knowledge/esm2_embeddings_d5120.pt',
    "esm_embedding_path": "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_pretrain/esm2_embeddings_d5120.pt",
    'pretrain_ckpt': None,
    'gene_id': True
}

config_train2 = {
    'pretrained_path': None,
    'retake_training': False,
    'dim_feedforward': 1536,
    'nheads': 16,
    'masking_p': 0.15,
    'nlayers': 18,
    'dropout': 0.1,
    'dim_model': 1536,
    'batch_first': True,
    'n_tokens': 92076,
    'n_connect_comp': 46714,
    'n_aux': 20,
    'n_rna_type': 33,
    'batch_size': 16,
    'context_length': 1000,
    'lr': 1e-05,
    'warmup': 10000,
    'max_epoch': 100,
    'total_steps': 100000,
    'autoregressive': False,
    'pool': None,
    'supervised_task': False,
    'learnable_pe': True,
    'organ': 'everything',
    'specie': True,
    'assay': True,
    'modality': False,
    'contrastive': False,
    'neighbor_enhance': True,
    'num_neighbors': 4,
    'use_esm_embedding': True,
    'ems_embedding_dim': 5120,
    'root_path': '/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST',
    'esm_embedding_path': '/raid/zhangchengming/BrainBeacon-master/prior_knowledge/esm2_embeddings_d5120.pt',
    'pretrain_ckpt': None,
    'gene_id': True
}

train_path = [
    # === public_tokenier_joblib_0314 ===
    # MERFISH - Human (207 job files, 3 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Human_Developing",
    # MERFISH - Mouse (14,436 job files, 551 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Mouse_Merfish_Visp",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Mouse_Yao2023Atlas",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Mouse_Zhang2023Amolecularly_rawcount",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Mouse_moffitt2018molecular",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/MERFISH/MERFISH_Mouse_zhang2021spatially",
    # STARmap - Mouse (585 job files, 38 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/STARmap/STARmap_Mouse_Shi2023Spatial",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/STARmap/STARmap_Mouse_Wang2018three",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/STARmap/STARmap_Mouse_Zeng2023Integrative",
    # SlideseqV2 - Mouse (1,836 job files, 48 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/SlideseqV2/SlideseqV2_Mouse_Cable2022CSIDE",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/SlideseqV2/SlideseqV2_Mouse_Wang2022OB",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/SlideseqV2/SlideseqV2_Mouse_stickels2020highly",
    # Xenium - Human (27 job files, 1 sample)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/Xenium/Xenium_Human_Preview_FFPE",
    # Xenium - Mouse (792 job files, 8 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/Xenium/Xenium_Mouse_Alzheimer_FFPE",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/Xenium/Xenium_Mouse_ExplorerDemo_FF",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/Xenium/Xenium_Mouse_Hemisphere_FF",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/public_tokenier_joblib_0314/Xenium/Xenium_Mouse_Replicates_FF",
    # === stereoseq_tokenier_joblib_0314 ===
    # Stereo-seq - Human (2,277 job files, 47 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/human/impu",
    # Stereo-seq - Macaque (83,214 job files, 159 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/macaqe1/impu",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/macaque3/impu",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/macque2/impu",
    # Stereo-seq - Marmoset (24,228 job files, 150 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/marmoset_1/impu",
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/marmoset_2/impu",
    # Stereo-seq - Mouse (7,578 job files, 195 samples)
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/stereoseq_tokenier_joblib_0314/mouse/impu",
]
