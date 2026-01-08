import os
from math import ceil
from typing import Dict, List

from torch.distributed import get_rank
import time

import pytorch_lightning as pl
import merlin.io
from merlin.dataloader.torch import Loader
from merlin.dtypes import boolean
from merlin.dtypes import float32, int64, int32
from merlin.schema import ColumnSchema, Schema


PARQUET_SCHEMA = {
    'X': int32,
    'X_connect_comp': int32,
    'X_neighbor_0': int32,
    'X_neighbor_0_connect_comp': int32,
    'X_neighbor_1': int32,
    'X_neighbor_1_connect_comp': int32,
    'X_neighbor_2': int32,
    'X_neighbor_2_connect_comp': int32,
    'X_neighbor_3': int32,
    'X_neighbor_3_connect_comp': int32,
    'X_niche_0': int32,
    'X_niche_1': int32,
    'X_niche_2': int32,
    'X_niche_3': int32,
    'X_niche_4': int32,
    'density_0': float32,
    'density_1': float32,
    'density_2': float32,
    'density_3': float32,
    'density_4': float32,
    'niche': int64,
    'author_cell_type': int64,
    'region': int64,
    'soma_joinid': int64,
    'is_primary_data': boolean,
    'dataset_id': int64,
    'donor_id': int64,
    'assay': int64,
    'cell_type': int64,
    'development_stage': int64,
    'disease': int64,
    'tissue': int64,
    'tissue_general': int64,
    'tech_sample': int64,
    'idx': int64,
    'specie': int64,
    'modality': int64,
    'organism': int64,
    'measured_genes': int32,
}


def merlin_dataset_factory(path: str, columns: List[str], dataset_kwargs: Dict[str, any]):
    x_schema = ColumnSchema(
        'X',
        dtype=PARQUET_SCHEMA['X'],
        is_list=True,
        # is_ragged=False,
        properties={'value_count': {'max': 2048}}
    )
    x_homo = ColumnSchema(
        'X_connect_comp',
        dtype=PARQUET_SCHEMA['X_connect_comp'],
        is_list=True,
        # is_ragged=False,
        # properties={'value_count': {'max': 2048}}
    )
    x_neighbor_0 = ColumnSchema(
        'X_neighbor_0',
        dtype=PARQUET_SCHEMA['X'],
        is_list=True,
    )
    x_neighbor_0_connect_comp = ColumnSchema(
        'X_neighbor_0_connect_comp',
        dtype=PARQUET_SCHEMA['X_neighbor_0_connect_comp'],
        is_list=True,
    )
    x_neighbor_1 = ColumnSchema(
        'X_neighbor_1',
        dtype=PARQUET_SCHEMA['X_neighbor_1'],
        is_list=True,
    )
    x_neighbor_1_connect_comp = ColumnSchema(
        'X_neighbor_1_connect_comp',
        dtype=PARQUET_SCHEMA['X_neighbor_1_connect_comp'],
        is_list=True,
    )
    x_neighbor_2 = ColumnSchema(
        'X_neighbor_2',
        dtype=PARQUET_SCHEMA['X_neighbor_2'],
        is_list=True,
    )
    x_neighbor_2_connect_comp = ColumnSchema(
        'X_neighbor_2_connect_comp',
        dtype=PARQUET_SCHEMA['X_neighbor_2_connect_comp'],
        is_list=True,
    )
    x_neighbor_3 = ColumnSchema(
        'X_neighbor_3',
        dtype=PARQUET_SCHEMA['X_neighbor_3'],
        is_list=True,
    )
    x_neighbor_3_connect_comp = ColumnSchema(
        'X_neighbor_3_connect_comp',
        dtype=PARQUET_SCHEMA['X_neighbor_3_connect_comp'],
        is_list=True,
    )
    return merlin.io.Dataset(
        path,
        engine='parquet',
        schema=Schema(
            [
                x_schema, x_homo,x_neighbor_0, x_neighbor_0_connect_comp,x_neighbor_1, x_neighbor_1_connect_comp,
                x_neighbor_2, x_neighbor_2_connect_comp, x_neighbor_3, x_neighbor_3_connect_comp
            ]
            +
            [ColumnSchema(col, dtype=PARQUET_SCHEMA[col]) for col in columns]
        ),
        cpu=True,
        **dataset_kwargs
    )


def set_default_kwargs_dataloader(kwargs: Dict[str, any] = None, training: bool = True):
    assert isinstance(training, bool)
    if kwargs is None:
        kwargs = {}
    if 'parts_per_chunk' not in kwargs:
        kwargs['parts_per_chunk'] = 8 if training else 1
    # if 'drop_last' not in kwargs:
    #     kwargs['drop_last'] = training
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = training

    return kwargs


def set_default_kwargs_dataset(kwargs: Dict[str, any] = None, training: bool = True):
    if kwargs is None:
        kwargs = {}
    if all(['part_size' not in kwargs, 'part_mem_fraction' not in kwargs]):
        kwargs['part_size'] = '300MB' if training else '300MB'

    return kwargs


def _get_data_files_distributed(base_path: list, world_size: int, sub_sample_frac: float = 1) -> List:
    files_devices = []
    all_files = []
    for path in base_path:
        all_files.extend([(path, f_path) for f_path in os.listdir(path) if f_path.endswith('.parquet')])

    for device in range(world_size):
        files = [(path, file_path) for (path, file_path) in all_files if int(file_path.split('.')[0].split('-')[1]) % world_size == device]
        files = [os.path.join(path, file_path) for (path, file_path) in sorted(files, key=lambda x: int(x[1].split('.')[0].split('-')[1]))]
        files.sort(reverse=True)
        files_devices.append(files[:ceil(sub_sample_frac * len(files))])
    return files_devices


def _create_single_distributed_dataset(
        files_devices: List[str],
        columns: List[str],
        world_size: int,
        dataset_kwargs_train: Dict[str, any] = None,
        training: bool = True
):
    datasets = []

    for device in range(world_size):
        dataset = merlin_dataset_factory(
            files_devices[device],
            columns,
            set_default_kwargs_dataset(dataset_kwargs_train, training=training)
        )
        datasets.append(dataset)

    return datasets


class MerlinDataModuleDistributed(pl.LightningDataModule):
    def __init__(
            self,
            train_path: list,
            eval_path: list,
            columns: List[str],
            batch_size: int,
            world_size: int,
            sub_sample_frac: float = 1.,
            dataloader_kwargs_train: Dict[str, any] = None,
            dataloader_kwargs_inference: Dict[str, any] = None,
            dataset_kwargs_train: Dict[str, any] = None
    ):
        super().__init__()
        for col in columns:
            assert col in PARQUET_SCHEMA
        self.columns = columns
        self.world_size = world_size
        self.batch_size = batch_size
        self.files_devices_train = _get_data_files_distributed(
            train_path, world_size=world_size, sub_sample_frac=sub_sample_frac
        )
        self.files_devices_val = _get_data_files_distributed(
            eval_path, world_size=world_size, sub_sample_frac=sub_sample_frac
        )
        self.files_devices_test = _get_data_files_distributed(
            eval_path, world_size=world_size, sub_sample_frac=sub_sample_frac
        )
        self.dataset_kwargs_train = dataset_kwargs_train

    # def setup(self, stage: str) -> None:
        self.dataloader_kwargs_train = set_default_kwargs_dataloader(training=True)
        self.dataloader_kwargs_inference = set_default_kwargs_dataloader(training=False)
        time1 = time.time()
        self.train_datasets = _create_single_distributed_dataset(
            files_devices=self.files_devices_train,
            columns=self.columns,
            world_size=self.world_size,
            training=False
        )
        time2 = time.time()
        print(f"train datasets created in {time2 - time1}s")

        time1 = time.time()
        self.val_datasets = _create_single_distributed_dataset(
            files_devices=self.files_devices_val,
            columns=self.columns,
            world_size=self.world_size,
            training=False
        )
        time2 = time.time()
        print(f"val datasets created in {time2 - time1}s")

        time1 = time.time()
        self.test_datasets = _create_single_distributed_dataset(
            files_devices=self.files_devices_test,
            columns=self.columns,
            world_size=self.world_size,
            training=False
        )
        time2 = time.time()
        print(f"test datasets created in {time2 - time1}s")

        self.prepare_data_per_node = True
        self._log_hyperparams = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    # def train_dataloader(self):
    #     return DataLoader(self.train_datasets[get_rank()], batch_size=self.batch_size)
    #
    # def val_dataloader(self):
    #     return DataLoader(self.val_datasets[get_rank()], batch_size=self.batch_size)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test_datasets[get_rank()], batch_size=self.batch_size)

    def train_dataloader(self):
        return Loader(self.train_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_train)

    def val_dataloader(self):
        return Loader(self.val_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def test_dataloader(self):
        return Loader(self.test_datasets[get_rank()], batch_size=self.batch_size, **self.dataloader_kwargs_inference)

    def predict_dataloader(self):
        return Loader(self.test_datasets, batch_size=self.batch_size, **self.dataloader_kwargs_inference)
