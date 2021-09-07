from random import getrandbits
from typing import Optional
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data.dataloader import T_co
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.dataloader import _collate_fn_t
from torch.utils.data.dataloader import _worker_init_fn_t


class OmniSpheroDataLoader(DataLoader):

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False,
                 sampler: Optional[Sampler[int]] = None, batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: _collate_fn_t = None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn: _worker_init_fn_t = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False, transform_enabled: bool = False, transform_data_saver: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

        self.transform_enabled = transform_enabled is not None
        self.untransformed_data = None
        self.transform_data_saver = transform_data_saver

        if self.transform_enabled:
            self.untransformed_data = dataset

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        base_iter: _BaseDataLoaderIter = super()._get_iterator()

        if self.transform_enabled:
            base_iter = self.augment_data_loader(base_iter)

        return base_iter

    def augment_data_loader(self, base_iter: _BaseDataLoaderIter):
        iter_ds = base_iter._dataset
        if not self.transform_data_saver:
            iter_ds = self.untransformed_data

        augmented_ds = []
        hash_list = []

        # Augmenting the dataset and writing it into a new list
        for i in range(len(iter_ds)):
            # print('Augmenting bag: ' + str(i))
            current_X = iter_ds[i][0]
            augmented_X = []
            hash_list.append(hash(current_X.tostring()))

            for j in range(current_X.shape[0]):
                current_sample = current_X[j]
                current_sample = np.copy(current_sample)

                r = current_sample[0]
                g = current_sample[1]
                b = current_sample[2]
                if bool(getrandbits(1)):
                    r = np.fliplr(r)
                    g = np.fliplr(g)
                    b = np.fliplr(b)
                if bool(getrandbits(1)):
                    r = np.flipud(r)
                    g = np.flipud(g)
                    b = np.flipud(b)

                augmented_sample = np.dstack((r, g, b))
                augmented_sample = np.copy(augmented_sample)
                augmented_X.append(augmented_sample)

            augmented_X = np.asarray(augmented_X)
            augmented_X = np.copy(augmented_X)
            augmented_X = np.einsum('bhwc->bchw', augmented_X)
            augmented_ds.append(augmented_X)

        # Applying the augmented list to the dataset
        for i in range(len(augmented_ds)):
            y = base_iter._dataset[i][1]
            y_samples = base_iter._dataset[i][2]
            X_raw = base_iter._dataset[i][3]

            base_iter._dataset[i] = (augmented_ds[i], y, y_samples, X_raw)

        debug_hash = hash(str(hash_list))
        print('Data loader: Saver: '+str(self.transform_data_saver)+'. Size: '+str(len(iter_ds))+'. Hash: '+str(debug_hash))

        return base_iter

    def __iter__(self) -> '_BaseDataLoaderIter':
        base_iter: _BaseDataLoaderIter = super().__iter__()
        return base_iter

    def __len__(self) -> int:
        return super().__len__()
