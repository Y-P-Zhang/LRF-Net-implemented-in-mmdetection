'''
Transform the official trained models into mmdetection version. 
'''

import torch
import mmcv
#import mmdet


from collections import OrderedDict
import time




lrf_vgg_coco_512 = 'LRF_vgg_COCO_512.pth'



def weights_to_cpu(state_dict):
    if type(state_dict) == dict:
        state_dict = state_dict['state_dict']
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu



def transform(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():  
        key = key[7:]
        if key.split('.')[0] == 'conf':
            name = 'bbox_head.cls_convs.'+ key.split('.')[1]\
                    + '.'+key.split('.')[2]
            new_state_dict[name] = value
        elif key.split('.')[0] == 'loc':
            name = 'bbox_head.reg_convs.'+ key.split('.')[1]\
                    + '.'+key.split('.')[2]
            new_state_dict[name] = value
        else:
            name = 'backbone.' + key
            new_state_dict[name] = value
    return new_state_dict



def save_checkpoint(state_dict, optimizer=None, meta=None):
    if meta is None:
        meta = {}
        
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))
    #meta.update(mmdet_version=mmdet.__version__, time=time.asctime(),
    #            mmcv_version=mmcv.__version__)
    meta.update(time=time.asctime(),
                mmcv_version=mmcv.__version__)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(state_dict)
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    filename = lrf_vgg_coco_512[:-4] + '_transform' + '.pth'

    torch.save(checkpoint, filename)




lrf_state_dict = torch.load(lrf_vgg_coco_512)
transformed = transform(lrf_state_dict)
save_checkpoint(transformed)


          
            





