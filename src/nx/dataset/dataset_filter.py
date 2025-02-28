import os
import copy
import enum
import pathlib
import logging
import json
from dataclasses import dataclass
import uuid

import pickle
import multiprocessing.shared_memory
import torch
import ultralytics.data.dataset

import nx.dataset.utils


logger = logging.getLogger(__name__)


class DatasetFilter(object):
    # Delegate all methods to wrapped dataset,
    # but override __getitem__, __getitems__ for return only enabled.
    # state of filter controlled over shared memory, because dataloader live in separate process.

    #class ConfigHolder(object):
        
    def __init__(self, dataset):
        self._dataset = dataset
        self._shared_config_name = str(uuid.uuid4())
        # < id of shared memory for publish changes, should be inherited after fork.
        self._shared_config_id = None
        self._shared_config = None
        self._config = None
        self._last_config_change_id = None  # < Optimization for avoid excess config deseralization.

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._close_shared_config()
        if self._shared_config_id is not None:
            self._shared_config_id.close()
            try:
                self._shared_config.unlink()
            except:
                pass
            self._shared_config_id = None

    def __getattr__(self, method_name):
        if self._is_delegate_method(method_name):
            return getattr(self._dataset, method_name)
        else:
            return object.__getattribute__(self, method_name)

    def __len__(self):
        return len(self._dataset)

    # Make object pickable (for fork in ultralytics dataloader process)
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_shared_config_id']
        del state['_shared_config']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shared_config_id = None
        self._shared_config = None

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indexes):
        # Preprocess indexes for return only enabled.
        self._update_config()
        use_indexes = indexes
        if self._config is not None:
            if "indexes" in self._config is not None and self._config["indexes"] is not None:
                # Map indexes to enabled only.
                use_indexes = []
                indexes_map = self._config["indexes"]
                l = len(indexes_map)
                for i in indexes:
                    use_indexes.append(indexes_map[i % l])
        return self._dataset.__getitems__(use_indexes)

    def enable_indexes(self, indexes):
        # Generate unique id for publish chages.
        change_id = str(uuid.uuid4())

        config = {}
        config['indexes'] = indexes
        serialized_config = pickle.dumps(config)

        # Create filled shared memory with config.
        shared_config = multiprocessing.shared_memory.SharedMemory(
            name=change_id + ".config",
            create=True,
            size=len(serialized_config)
        )
        shared_config.buf[:] = serialized_config
        
        if self._shared_config_id is None:
            self._shared_config_id = multiprocessing.shared_memory.SharedMemory(
                name=self._shared_config_name + ".id",
                create=True,
                size=len(change_id),
            )

        self._close_shared_config()
        self._shared_config = shared_config
        self._shared_config_id.buf[:] = change_id.encode('utf-8')  # < Publish config id.

    def _close_shared_config(self):
        if self._shared_config is not None:
            self._shared_config.close()
            self._shared_config = None

    def _update_config(self):
        if self._shared_config_id is None:
            try:
                self._shared_config_id = multiprocessing.shared_memory.SharedMemory(
                    name=self._shared_config_name + ".id"
                )
            except FileNotFoundError:
                # config isn't published yet
                return

        # read config id.
        while True:
            change_id = self._shared_config_id.buf.tobytes().decode('utf-8')
            if self._last_config_change_id is not None and change_id == self._last_config_change_id:
                return
            # Try read change.
            try:
                shared_config = multiprocessing.shared_memory.SharedMemory(
                    name=change_id + ".config",
                )
                # read config
                self._config = pickle.loads(shared_config.buf)
                self._last_config_change_id = change_id
                shared_config.close()
                shared_config = None
                return
            except FileNotFoundError:
                # config isn't published yet
                time.sleep(1)

    def _is_delegate_method(self, method_name):
        return method_name not in [
            '__getitem__',
            '__getitems__',
            # Members
            '_shared_config_name',
            '_shared_config_id',
            '_config',
            '_last_config_change_id',
            # Methods
            '_is_delegate_method',
            '_update_config',
            'enable_indexes',
            '__getstate__',
            '__getstate__',
        ]
