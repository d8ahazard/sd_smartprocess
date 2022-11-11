import _codecs
import collections
import pickle

import numpy
import torch

from modules import safe
from modules.safe import TypedStorage


def encode(*args):
    out = _codecs.encode(*args)
    return out


class RestrictedUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        assert saved_id[0] == 'storage'
        return TypedStorage()

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return getattr(collections, name)
        if module == 'torch._utils' and name in ['_rebuild_tensor_v2', '_rebuild_parameter']:
            return getattr(torch._utils, name)
        if module == 'torch' and name in ['FloatStorage', 'HalfStorage', 'IntStorage', 'LongStorage', 'DoubleStorage',
                                          'ByteStorage']:
            return getattr(torch, name)
        if module == 'torch.nn.modules.container' and name in ['ParameterDict', 'Sequential']:
            return getattr(torch.nn.modules.container, name)
        if module == 'numpy.core.multiarray' and name == 'scalar':
            return numpy.core.multiarray.scalar
        if module == 'numpy' and name == 'dtype':
            return numpy.dtype
        if module == '_codecs' and name == 'encode':
            return encode
        if module == "pytorch_lightning.callbacks" and name == 'model_checkpoint':
            import pytorch_lightning.callbacks
            return pytorch_lightning.callbacks.model_checkpoint
        if module == "pytorch_lightning.callbacks.model_checkpoint" and name == 'ModelCheckpoint':
            import pytorch_lightning.callbacks.model_checkpoint
            return pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint
        if "yolo" in module:
            return super().find_class(module, name)
        if module == "models.common" and name == "Conv":
            return super().find_class(module, name)
        if 'torch.nn.modules' in module and name in ['Conv', 'Conv2d', 'BatchNorm2d', "SiLU", "MaxPool2d", "Upsample",
                                                     "ModuleList"]:
            return super().find_class(module, name)
        if "models.common" in module and name in ["C3", "Bottleneck", "SPPF", "Concat"]:
            return super().find_class(module, name)
        if module == "__builtin__" and name == 'set':
            return set

        # Forbid everything else.
        raise pickle.UnpicklingError(f"global '{module}/{name}' is forbidden")


safe.RestrictedUnpickler = RestrictedUnpickler
