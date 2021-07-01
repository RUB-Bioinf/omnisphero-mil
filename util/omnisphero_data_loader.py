from typing import Optional
from typing import Optional
from typing import Optional
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import Sampler
from torch.utils.data.dataloader import T_co
from torch.utils.data.dataloader import _collate_fn_t
from torch.utils.data.dataloader import _worker_init_fn_t


class OmniSpheroDataLoader(DataLoader):

    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: bool = False,
                 sampler: Optional[Sampler[int]] = None, batch_sampler: Optional[Sampler[Sequence[int]]] = None,
                 num_workers: int = 0, collate_fn: _collate_fn_t = None, pin_memory: bool = False,
                 drop_last: bool = False, timeout: float = 0, worker_init_fn: _worker_init_fn_t = None,
                 multiprocessing_context=None, generator=None, *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)

