"""Author-specific training inputs and the original shared evaluation targets."""
import hashlib
import random
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from skimage import io
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

from recipes import ASSETS, ROOT, POLICY, adapters, definitions

DATA = ROOT.parent/'datasets/Kvasir-SEG'
SPLITS = ROOT/'experiments/exp00_baseline/configs/splits/kvasir'
MEAN,STD = (.485,.456,.406),(.229,.224,.225)


def split_ids():
    result = {name:(SPLITS/f'{name}.txt').read_text().splitlines() for name in ('train','val','test')}
    if len(result['train'])!=880 or len(result['val'])!=120 or result['val']!=result['test']:
        raise RuntimeError('Original880/120/shared-test split changed')
    if len(set(result['train']))!=880 or set(result['train'])&set(result['val']):
        raise RuntimeError('Duplicate or leaked split')
    return result


def seed_for(seed, epoch, index):
    return int.from_bytes(hashlib.sha256(f'paper-native-v1:{seed}:{epoch}:{index}'.encode()).digest()[:4],'little')


def edge_from_author(mask):
    """Supplied edge_util.py, with in-memory equivalent of cv2.imwrite rounding."""
    body = cv2.distanceTransform(mask,cv2.DIST_L2,5)**.5
    if len(body[body>0]):
        body[body>2] = 255
    result = cv2.blur(mask-body,(3,3))
    return np.clip(np.rint(result),0,255).astype(np.uint8)


class NativeDataset(Dataset):
    def __init__(self,name,training,seed,stage=2,ids=None):
        self.name,self.training,self.seed,self.stage = name,training,seed,stage
        self.ids = list(ids if ids is not None else split_ids()['train' if training else 'val'])
        self.epoch = 0
        self.transform = None
        self.native = None
        self.image_transform = T.Compose([T.Resize((352,352)),T.ToTensor(),T.Normalize(MEAN,STD)])
        self.mask_transform = T.Compose([T.Resize((352,352)),T.ToTensor()])
        if training and name == 'polyper':
            adapters.setup_dependencies()
            import mmcv
            from mmcv.transforms import RandomResize,RandomFlip
            from mmcv.transforms.base import BaseTransform
            from mmcv.transforms.utils import cache_randomness
            path = ASSETS/'vendor/polyper/mmsegmentation/mmseg/datasets/transforms/transforms.py'
            classes = definitions(path,{'RandomCrop','PhotoMetricDistortion'},dict(
                np=np,random=np.random,mmcv=mmcv,BaseTransform=BaseTransform,
                cache_randomness=cache_randomness,Sequence=Sequence,Union=Union,Tuple=Tuple))
            self.transform = (RandomResize(scale=(512,512),ratio_range=(.5,2.),keep_ratio=True),
                classes['RandomCrop'](crop_size=(224,224),cat_max_ratio=.75),
                RandomFlip(prob=.5),classes['PhotoMetricDistortion']())
        elif training and name in ('eat','condseg'):
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            if name == 'eat':
                self.transform = A.Compose([A.Normalize(),A.Resize(352,352,interpolation=cv2.INTER_NEAREST),
                    A.HorizontalFlip(p=.2),A.VerticalFlip(p=.2),ToTensorV2()])
            else:
                angle,p = (90,.5) if stage==1 else (35,.3)
                self.transform = A.Compose([A.Rotate(limit=angle,p=p),A.HorizontalFlip(p=p),A.VerticalFlip(p=p),
                    A.CoarseDropout(p=.3,max_holes=10,max_height=32,max_width=32)],
                    additional_targets={'background':'mask'})
        elif training and name == 'asgnet':
            path = ASSETS/'vendor/asgnet/ASGNet_26_TCSVT/utils/data_val.py'
            from PIL import ImageEnhance
            self.native = definitions(path,{'cv_random_flip','randomCrop','randomRotation','colorEnhance','randomPeper'},
                dict(Image=Image,ImageEnhance=ImageEnhance,random=random,np=np))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        # A continuous Polyper stream may repeat an index; occurrence is included
        # in the augmentation seed so repeated draws do not share a transform.
        occurrence = 0
        if isinstance(index,tuple):
            index,occurrence = index
        sample = self.ids[index]
        seed = seed_for(self.seed,self.epoch,index+len(self.ids)*occurrence)
        random.seed(seed);np.random.seed(seed)
        if self.transform is not None and hasattr(self.transform,'set_random_state'):
            self.transform.set_random_state(np.random.default_rng(seed),random.Random(seed))
        image_path,mask_path = DATA/'images'/sample,DATA/'masks'/sample
        image = cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(sample)
        background,edge = np.zeros((352,352),np.float32),np.zeros((352,352),np.float32)
        if not self.training:
            if self.name=='condseg':
                image = cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(352,352)).transpose(2,0,1)/255.
            elif self.name=='polyper':
                rgb=cv2.cvtColor(cv2.resize(image,(352,352)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
                image=((rgb-np.array(MEAN))/np.array(STD)).transpose(2,0,1)
            else:
                with Image.open(image_path) as im:
                    image=self.image_transform(im.convert('RGB'))
            # EXACT original reader and threshold, including nonzero JPEG values.
            mask=(io.imread(mask_path,as_gray=True)>0).astype(np.float32)
            mask=cv2.resize(mask,(352,352),interpolation=cv2.INTER_NEAREST)
            background=1-mask
        elif self.name=='polyper':
            mask=(mask>127).astype(np.uint8)  # OpenMMLab expects prepared class IDs0/1.
            results=dict(img=image,gt_seg_map=mask,seg_fields=['gt_seg_map'])
            for transform in self.transform:
                results=transform(results)
            image=cv2.cvtColor(cv2.resize(results['img'],(352,352)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            image=((image-np.array(MEAN))/np.array(STD)).transpose(2,0,1)
            mask=cv2.resize(results['gt_seg_map'],(352,352),interpolation=cv2.INTER_NEAREST)
        elif self.name=='eat':
            # Kvasir names are nonnumeric: official same-group index pairing is
            # the same image; preserve the LAB round trip and normalization.
            lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            numeric=[n for n in self.ids if n[:-4].isdigit()]
            other=[n for n in self.ids if not n[:-4].isdigit()]
            if other and np.random.rand()<.7:
                partner=other[index%len(other)]
            elif numeric:
                partner=numeric[index%len(numeric)]
            else:
                partner=sample
            lab2=cv2.cvtColor(cv2.imread(str(DATA/'images'/partner)),cv2.COLOR_BGR2LAB)
            mean,std=lab.mean((0,1),keepdims=True),lab.std((0,1),keepdims=True)
            if (std==0).any():
                raise ValueError(f'Undefined author color transfer: {sample}')
            transferred=np.uint8((lab-mean)/std*lab2.std((0,1),keepdims=True)+lab2.mean((0,1),keepdims=True))
            result=self.transform(image=cv2.cvtColor(transferred,cv2.COLOR_LAB2RGB),mask=mask/255.)
            image,mask=result['image'],result['mask']
        elif self.name=='condseg':
            result=self.transform(image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB),mask=mask,background=255-mask)
            image=cv2.resize(result['image'],(352,352)).transpose(2,0,1)/255.
            mask=cv2.resize(result['mask'],(352,352))/255.
            background=cv2.resize(result['background'],(352,352))/255.
        else:
            edge=cv2.dilate(edge_from_author(mask),np.ones((3,3),np.uint8),iterations=1)
            image,mask,edge=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),Image.fromarray(mask),Image.fromarray(edge)
            for key in ('cv_random_flip','randomCrop','randomRotation'):
                image,mask,edge=self.native[key](image,mask,edge)
            image=self.image_transform(self.native['colorEnhance'](image))
            mask=self.mask_transform(self.native['randomPeper'](mask)).squeeze(0)
            edge=(self.mask_transform(self.native['randomPeper'](edge)).squeeze(0)>0).float()
        def tensor(value):
            return value.float() if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value,dtype=np.float32))
        image,mask,background,edge=map(tensor,(image,mask,background,edge))
        if image.shape!=(3,352,352) or mask.shape!=(352,352):
            raise RuntimeError(f'Input shape mismatch: {sample}')
        return dict(image=image,mask=mask[None],background=background[None],edge=edge[None],id=sample)


class ContinuousEpochSampler(Sampler):
    def __init__(self,seed,size=880,batch=6):
        self.generator=torch.Generator().manual_seed(seed)
        self.size,self.draws=size,math_ceil(size/batch)*batch
        self.pending=[]
        self.occurrence=0

    def __len__(self):
        return self.draws

    def __iter__(self):
        result=[]
        while len(result)<self.draws:
            if not self.pending:
                self.pending=torch.randperm(self.size,generator=self.generator).tolist()
            count=min(self.draws-len(result),len(self.pending))
            result.extend(self.pending[:count]);del self.pending[:count]
        for index in result:
            self.occurrence+=1
            yield index,self.occurrence


def math_ceil(value):
    return int(np.ceil(value))


def loader(name,training,seed,stage=2,workers=2):
    dataset=NativeDataset(name,training,seed,stage)
    sampler=ContinuousEpochSampler(seed) if name=='polyper' and training else None
    batch=POLICY[name]['batch'] if training else (4 if name=='condseg' else 1)
    return DataLoader(dataset,batch_size=batch,shuffle=training and sampler is None,
        sampler=sampler,num_workers=workers,pin_memory=True,persistent_workers=False,
        generator=torch.Generator().manual_seed(seed))
