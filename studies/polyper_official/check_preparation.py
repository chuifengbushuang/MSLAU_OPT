"""Check actual author config inheritance and real annotation decoding on CPU."""
import io
import json
import zipfile

import cv2
import numpy as np
from mmengine.config import Config
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules

from prepare import ROOT, WORK, OUT, BASE, SIZES, sha, write


def main():
    register_all_modules()
    base, cfg = Config.fromfile(BASE), Config.fromfile(WORK / 'official_80k.py')
    assert cfg.train_cfg == base.train_cfg
    assert cfg.optim_wrapper == base.optim_wrapper
    assert cfg.param_scheduler == base.param_scheduler
    assert cfg.train_pipeline == base.train_pipeline
    assert cfg.test_pipeline == base.test_pipeline
    assert cfg.model.decode_head == base.model.decode_head
    assert cfg.model.data_preprocessor == base.model.data_preprocessor
    backbone = dict(cfg.model.backbone)
    backbone['init_cfg'] = base.model.backbone.init_cfg
    assert backbone == base.model.backbone
    assert cfg.train_dataloader.batch_size == base.train_dataloader.batch_size == 6
    assert cfg.train_dataloader.sampler == base.train_dataloader.sampler
    assert cfg.train_dataloader.dataset.times == 40000
    assert cfg.val_evaluator.type == base.val_evaluator.type == 'IoUMetric'
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    assert len(dataset) == 100
    checked = []
    with zipfile.ZipFile(WORK / 'downloads/TestDataset.zip') as z:
        for i in (0, 49, 99):
            sample = dataset[i]
            data = sample['data_samples']
            name = data.img_path.rsplit('/', 1)[-1]
            original = cv2.imdecode(np.frombuffer(z.read('TestDataset/Kvasir/masks/' + name),
                                      np.uint8), cv2.IMREAD_UNCHANGED) > 0
            loaded = data.gt_sem_seg.data[0].numpy()
            assert np.array_equal(loaded, original.astype(np.uint8))
            assert set(np.unique(loaded)) == {0, 1}
            assert loaded.shape == data.ori_shape
            assert max(sample['inputs'].shape[1:]) == 512
            checked.append(dict(id=name, input_shape=list(sample['inputs'].shape),
                                gt_shape=list(loaded.shape), foreground_pixels=int(loaded.sum())))
    write(OUT / 'preparation_check.json', dict(status='passed', checked_samples=checked,
          all_training_math_config_equal_to_released_main=True,
          native_gt_geometry_and_class_ids_verified=True,
          source_manifest_sha256=sha(OUT / 'source_manifest.json')))
    print(json.dumps(dict(status='passed', checked=checked)))


if __name__ == '__main__':
    main()
