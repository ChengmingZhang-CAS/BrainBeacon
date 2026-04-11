import numpy as np
import torch
import logging
import yaml
from brainbeacon.brain_beacon import BrainBeacon
from brainbeacon.brain_beacon import PretrainJoblibDataset
from brainbeacon.brain_beacon import train_one_epoch
from brainbeacon.brain_beacon import save_checkpoint
from brainbeacon.sampler import HierarchicalDistributedSampler
from brainbeacon.configs.config import specie_dict, technology_dict
from torch.utils.data import DataLoader
from brainbeacon.configs.config_train import config_train as default_config_train
from brainbeacon.configs.config_train import train_path
from brainbeacon.configs.config_train import platform_prob
from brainbeacon.configs.config_train import species_prob
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import time
from torch.utils.tensorboard import SummaryWriter
from brainbeacon.brain_beacon import get_linear_warmup_scheduler


def load_config(config_path=None):
    """加载配置：指定 YAML 则从文件读取，否则使用 config_train.py 中的默认配置。"""
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    return dict(default_config_train)


seed = 2025
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main(rank, args, config_train):
    print('-'*10, rank, args, '-'*10)
    local_rank = rank

    exp_name = getattr(args, 'exp_name', None)
    logdir = f"{exp_name}_{int(time.time())}" if exp_name else f"{int(time.time())}"
    if rank == 0:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'tensorboard'), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(logdir, 'tensorboard'))
        """Set up logging for rank 0 only."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(logdir, "training.log")),
                logging.StreamHandler()  # Optional: Print to console
            ]
        )
        logger = logging.getLogger(__name__)
    else:
        writer = None
        logger = None
    # Set up distributed training environment
    torch.distributed.init_process_group(backend='nccl')
    device = torch.device(f'cuda:{local_rank}')
    # Model, dataset, dataloader, and optimizer setup
    config_train['batch_size'] = 1
    model = BrainBeacon(
        dim_model=config_train["dim_model"],
        nheads=config_train['nheads'],
        dim_feedforward=config_train['dim_feedforward'],
        nlayers=config_train['nlayers'],
        dropout=config_train['dropout'],
        n_tokens=config_train["n_tokens"],
        n_connect_comp=config_train["n_connect_comp"],
        n_aux=config_train["n_aux"],
        n_rna_type=config_train['n_rna_type'],
        n_neighbor=config_train['num_neighbors'],
        esm_embedding_dim=config_train['ems_embedding_dim'],
        total_context_length=config_train['context_length'] * config_train['num_neighbors'],
        # Ablation study switches (from argparse)
        neighbor_enhance=bool(args.neighbor_enhance),
        use_gene_id_emb=bool(args.use_gene_id_emb),
        use_homo_emb=bool(args.use_homo_emb),
        use_rna_type_emb=bool(args.use_rna_type_emb),
        use_esm_emb=bool(args.use_esm_emb),
        use_pos_emb=bool(args.use_pos_emb),
        use_density_emb=bool(args.use_density_emb),
        density_token_idx=args.density_token_idx
    )

    # platform species probability
    label_weights = {(p, s): 1 / (pp * sp) for p, pp in platform_prob.items() for s, sp in species_prob.items()}
    if rank == 0:
        logger.info(f'Label weights: {label_weights}')

    # Non-strict checkpoint loading: skip shape-mismatched params (e.g. homo_connect_embedding)
    if config_train['pretrain_ckpt']:
        print(f"load ckpt from {config_train['pretrain_ckpt']}")
        ckpt = torch.load(config_train['pretrain_ckpt'], map_location='cpu')
        pretrained_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model_dict = model.state_dict()
        compatible_dict = {}
        skipped = []
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            elif k in model_dict:
                skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_dict[k].shape)}")
            else:
                skipped.append(f"{k}: not in model")
        if skipped:
            print(f"[Warning] Skipped {len(skipped)} params during ckpt loading:")
            for s in skipped:
                print(f"  {s}")
        print(f"[INFO] Loaded {len(compatible_dict)}/{len(model_dict)} params from ckpt")
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        # Don't restore optimizer if shapes changed (e.g. n_connect_comp mismatch)
        if skipped:
            print("[INFO] Shape mismatch detected, skipping optimizer state restore")
            ckpt = None
    else:
        ckpt = None
    model = model.to(device)
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.numel()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))
    print(f'param count: {param_count / 1000000000:.4f}B')

    if rank == 0:
        logger.info(f"Model size: {size_all_mb:.3f} MB")
        logger.info(f"param count: {param_count / 1000000000:.4f}B")
        logger.info(model)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        logger.info(f"args: {args}")
        logger.info(f"config_train: {config_train}")
        logger.info(f"train path: {train_path}")

    if rank == 0:
        logger.info("Preparing datasets...")
    masked_indices_files_list = []
    mask_files_list = []
    real_indices_files_list = []
    attention_mask_files_list = []
    connect_comp_files_list = []
    rna_type_files_list = []
    neighbor_gene_distribution_files_list = []
    file_prefix_list = []
    file_labels_list = []
    for path in train_path:
        # Parse (platform_id, species_id) from directory name
        dirname = os.path.basename(path).lower()
        platform_id = 11  # default to stereo
        for tech_name, tech_id in technology_dict.items():
            if tech_name.lower() in dirname:
                platform_id = tech_id
                break
        species_id = 4  # default to mouse
        for spec_name, spec_id in specie_dict.items():
            if spec_name in dirname:
                species_id = spec_id
                break
        path_label = (platform_id, species_id)

        for sample in os.listdir(path):
            sample_dir = os.path.join(path, sample)
            if not os.path.isdir(sample_dir):
                continue
            for prefix in os.listdir(sample_dir):
                prefix_dir = os.path.join(sample_dir, prefix)
                if not os.path.isdir(prefix_dir):
                    continue
                if len(os.listdir(prefix_dir)) >= 7:
                    file_prefix_list.append(prefix_dir)
                    file_labels_list.append(path_label)
                    for file in os.listdir(prefix_dir):
                        if 'masked_indices_' in file:
                            masked_indices_files_list.append(os.path.join(prefix_dir, file))
                        elif 'real_indices_' in file:
                            real_indices_files_list.append(os.path.join(prefix_dir, file))
                        elif 'attention_mask_' in file:
                            attention_mask_files_list.append(os.path.join(prefix_dir, file))
                        elif 'connect_comp_' in file:
                            connect_comp_files_list.append(os.path.join(prefix_dir, file))
                        elif 'rna_type_' in file:
                            rna_type_files_list.append(os.path.join(prefix_dir, file))
                        elif 'mask_' in file:
                            mask_files_list.append(os.path.join(prefix_dir, file))
                        elif 'neighbor_gene_distribution_' in file:
                            neighbor_gene_distribution_files_list.append(os.path.join(prefix_dir, file))
    if rank == 0:
        logger.info(f"Loaded {len(masked_indices_files_list)} files for training.")

    idx_list = [i for i in range(len(masked_indices_files_list))]
    random.shuffle(idx_list)
    # train_idx = [0]
    # val_idx = [0]
    train_idx = idx_list[:int(len(idx_list) * 0.95)]
    val_idx = idx_list[int(len(idx_list) * 0.95):]

    # 数据量消融：截断训练集到指定上限
    if args.max_total_samples is not None and args.max_total_samples < len(train_idx):
        train_idx = train_idx[:args.max_total_samples]
    train_dataset = PretrainJoblibDataset(
        [masked_indices_files_list[idx] for idx in train_idx],
        [mask_files_list[idx] for idx in train_idx],
        [real_indices_files_list[idx] for idx in train_idx],
        [attention_mask_files_list[idx] for idx in train_idx],
        [connect_comp_files_list[idx] for idx in train_idx],
        [rna_type_files_list[idx] for idx in train_idx],
        [neighbor_gene_distribution_files_list[idx] for idx in train_idx],
        file_prefix_list,
        file_labels=[file_labels_list[idx] for idx in train_idx]
    )
    valid_dataset = PretrainJoblibDataset(
        [masked_indices_files_list[idx] for idx in val_idx],
        [mask_files_list[idx] for idx in val_idx],
        [real_indices_files_list[idx] for idx in val_idx],
        [attention_mask_files_list[idx] for idx in val_idx],
        [connect_comp_files_list[idx] for idx in val_idx],
        [rna_type_files_list[idx] for idx in val_idx],
        [neighbor_gene_distribution_files_list[idx] for idx in val_idx],
        file_prefix_list,
        file_labels=[file_labels_list[idx] for idx in val_idx]
    )

    if rank == 0:
        logger.info(f"Training on {len(train_idx)} files and validating on {len(val_idx)} files.")

    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # val_sampler = DistributedSampler(valid_dataset, shuffle=False)

    desired_per_gpu_batch_count = len(train_dataset) // args.world_size

    train_sampler = HierarchicalDistributedSampler(
        dataset=train_dataset,
        label_fn=lambda i: train_dataset.get_label(i),
        label_weights=label_weights,
        num_samples=desired_per_gpu_batch_count,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True
    )

    # train_sampler = DistributedWeightedSampler(
    #     dataset=train_dataset,
    #     label_fn=lambda i: train_dataset.get_label(i),
    #     label_weights=label_weights,
    #     shuffle=True
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_train["batch_size"],
        sampler=train_sampler,
        num_workers=2,
        prefetch_factor=4
    )
    esm_embedding_map = torch.load(config_train["esm_embedding_path"], weights_only=False).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config_train["lr"])
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt['global_step']
    else:
        global_step = 0
    warmup_steps = config_train['warmup']  # 可以根据 batch_size 和数据大小调整
    lr_scheduler = get_linear_warmup_scheduler(optimizer, num_warmup_steps=warmup_steps)
    for epoch in range(config_train["max_epoch"]):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            logger.info(f"Starting epoch {epoch + 1}")
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device, rank, writer, esm_embedding_map, global_step, logger,
            logdir, epoch, lr_scheduler, max_steps=args.max_steps, accumulation_steps=args.accumulation_steps
        )
        # val_loss = validate(model, val_loader, criterion, device)
        # print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if rank == 0:
            # logger.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            writer.add_scalar('Loss/Train', train_loss, epoch)
            # writer.add_scalar('Loss/Validation', val_loss, epoch)
            save_checkpoint(epoch + 1, None, model.module, optimizer, os.path.join(logdir, f"epoch_{epoch + 1}.pt"))

        if args.max_steps is not None and global_step >= args.max_steps:
            if rank == 0:
                logger.info(f"Reached max_steps={args.max_steps}, stopping training.")
            break

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node', type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--node_rank', type=int)
    parser.add_argument('--nnode', type=int)
    # Ablation study switches (default: all enabled)
    parser.add_argument('--use_gene_id_emb', type=int, default=1, help='Use gene ID embedding (0/1)')
    parser.add_argument('--use_homo_emb', type=int, default=1, help='Use homologous connect component embedding (0/1)')
    parser.add_argument('--use_rna_type_emb', type=int, default=1, help='Use RNA type embedding (0/1)')
    parser.add_argument('--use_esm_emb', type=int, default=1, help='Use ESM protein embedding (0/1)')
    parser.add_argument('--use_pos_emb', type=int, default=1, help='Use positional embedding (0/1)')
    parser.add_argument('--neighbor_enhance', type=int, default=1, help='Use neighbor gene distribution embedding (0/1)')
    parser.add_argument('--use_density_emb', type=int, default=1, help='Use density token embedding (0/1)')
    parser.add_argument('--density_token_idx', type=int, default=2, help='Density token position index')
    parser.add_argument('--max_total_samples', type=int, default=None, help='Maximum total training samples for data ablation')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum total training steps (None means no limit)')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps (default=4 for continue training)')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name prefix for logdir (e.g. ABL1)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file (overrides config_train.py)')
    args = parser.parse_args()
    print('Args: {}'.format(args))

    config_train = load_config(args.config)
    print('Config: {}'.format(config_train))

    # set_seed(2025)
    # torch.multiprocessing.spawn(main, args=(args, ), nprocs=args.world_size * args.nproc_per_node, join=True)
    # torch.multiprocessing.spawn(main, args=(args,), nprocs=args.nproc_per_node)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank or 0))
    main(local_rank, args, config_train)
