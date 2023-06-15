import torch.nn as nn
import torch
from pathlib import Path
import numpy as np
import torchvision.transforms as T
import gdown
from os.path import exists as file_exists


from boxmot.utils.checks import TestRequirements
tr = TestRequirements()
from boxmot.utils import logger as LOGGER
from boxmot.deep.models import build_model

import time
import sys
from collections import OrderedDict
from boxmot.utils import logger as LOGGER


def check_suffix(file='osnet_ain_x1_0_msmt17.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                try:
                    assert s in suffix
                except AssertionError as err:
                    LOGGER.error(f"{err}{f} acceptable suffix is {suffix}")

def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    LOGGER.info('* url="{}"'.format(url))
    LOGGER.info('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    
    if not torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(weight_path)
        
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        LOGGER.warning(
            f'The pretrained weights "{weight_path}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'
        )
    else:
        LOGGER.success(
            f'Successfully loaded pretrained weights from "{weight_path}"'
        )
        if len(discarded_layers) > 0:
            LOGGER.warning(
                'The following layers are discarded '
                f'due to unmatched keys or layer size: {*discarded_layers,}'
            )


class ReIDDetectMultiBackend(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights='osnet_ain_x1_0_msmt17.pt', device=torch.device('cpu'), fp16=False):
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        self.pt = True
        self.fp16 = fp16
        self.fp16 &= self.pt or self.jit or self.engine  # FP16

        # Build transform functions
        self.device = device
        self.image_size=(256, 128)
        self.pixel_mean=[0.485, 0.456, 0.406]
        self.pixel_std=[0.229, 0.224, 0.225]
        self.transforms = []
        self.transforms += [T.Resize(self.image_size)]
        self.transforms += [T.ToTensor()]
        self.transforms += [T.Normalize(mean=self.pixel_mean, std=self.pixel_std)]
        self.preprocess = T.Compose(self.transforms)
        self.to_pil = T.ToPILImage()

        model_name = get_model_name(w)

        if w.suffix == '.pt':
            model_url = 'https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal' #'osnet_ain_x1_0_msmt17.pt'
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass

        # Build model
        self.model = build_model(
            model_name,
            num_classes=1,
            pretrained=not (w and w.is_file()),
            use_gpu=device
        )

        if self.pt:  # PyTorch
            # populate model arch with weights
            if w and w.is_file() and w.suffix == '.pt':
                load_pretrained_weights(self.model, w) 
            self.model.to(device).eval()
            self.model.half() if self.fp16 else  self.model.float()
        
        else:
            LOGGER.error('This model framework is not supported yet!')
            exit()

    def _preprocess(self, im_batch):

        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.to(self.device)

        return images
    
    
    def forward(self, im_batch):
        
        # preprocess batch
        im_batch = self._preprocess(im_batch)

        # batch to half
        if self.fp16 and im_batch.dtype != torch.float16:
           im_batch = im_batch.half()

        # batch processing
        features = []
        if self.pt:
            features = self.model(im_batch)
        elif self.jit:  # TorchScript
            features = self.model(im_batch)
        elif self.onnx:  # ONNX Runtime
            im_batch = im_batch.cpu().numpy()  # torch to numpy
            features = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im_batch})[0]
        elif self.engine:  # TensorRT
            if True and im_batch.shape != self.bindings['images'].shape:
                i_in, i_out = (self.model_.get_binding_index(x) for x in ('images', 'output'))
                self.context.set_binding_shape(i_in, im_batch.shape)  # reshape if dynamic
                self.bindings['images'] = self.bindings['images']._replace(shape=im_batch.shape)
                self.bindings['output'].data.resize_(tuple(self.context.get_binding_shape(i_out)))
            s = self.bindings['images'].shape
            assert im_batch.shape == s, f"input size {im_batch.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs['images'] = int(im_batch.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            features = self.bindings['output'].data
        elif self.xml:  # OpenVINO
            im_batch = im_batch.cpu().numpy()  # FP32
            features = self.executable_network([im_batch])[self.output_layer]
        else:
            LOGGER.error('Framework not supported at the moment, leave an enhancement suggestion')
            exit()

        if isinstance(features, (list, tuple)):
            return self.from_numpy(features[0]) if len(features) == 1 else [self.from_numpy(x) for x in features]
        else:
            return self.from_numpy(features)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=[(256, 128, 3)]):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.tflite
        if any(warmup_types) and self.device.type != 'cpu':
            im = [np.empty(*imgsz).astype(np.uint8)]  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup