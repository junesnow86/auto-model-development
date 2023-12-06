import warnings

warnings.filterwarnings('ignore')

import gc
import os
import sys

import mmcv
import mmcv.cnn as mmcv_cnn
import mmdet.core as mmdet_core
import torch
from mmcv.parallel import collate
from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

sys.path.append('/home/yileiyang/workspace/DNNGen/concrete_trace_test/nni')
sys.path.append('/home/yileiyang/workspace/DNNGen/concrete_trace_test')
from dnngen_utils.fxgraph_to_seq import Sequence, fold_seq
from nni.common.concrete_trace_utils import concrete_trace


def check_equal(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for sub_a, sub_b in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        keys_a, kes_b = set(a.keys()), set(b.keys())
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        # may not euqal on gpu
        return torch.std(a - b).item() < 1e-6
    else:
        return a == b
    
def check_nan(item):
    if isinstance(item, torch.Tensor):
        return torch.isnan(item).any()
    elif isinstance(item, (list, tuple, set)):
        for sub_item in item:
            if check_nan(sub_item):
                return True
        return False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(config_name):
    mmdet_dir = "/home/yileiyang/workspace/DNNGen/concrete_trace_test/mmdetection"
    config_dir = os.path.join(mmdet_dir, "configs")
    config_path = os.path.join(config_dir, config_name+".py")
    config = mmcv.Config.fromfile(config_path)

    def roi_align_setter(config_dict: dict):
        if 'type' in config_dict:
            if config_dict['type'] == 'RoIAlign':
                config_dict['use_torchvision'] = True
                config_dict['aligned'] = False
            else:
                for v in config_dict.values():
                    if isinstance(v, dict):
                        roi_align_setter(v)
    roi_align_setter(config._cfg_dict['model'])

    model = init_detector(config, device=device)
    model.eval()
    return model

def build_input(model):
    img = f"/home/yileiyang/workspace/DNNGen/concrete_trace_test/mmdetection/tests/data/color.jpg"
    config = model.cfg
    with torch.no_grad():
        config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)
        test_pipeline = Compose(config.data.test.pipeline)
        img_data = test_pipeline(dict(img_info=dict(filename=img), img_prefix=None))['img']
        img_tensor = collate(img_data, 1).data[0].to(device)
    return img_tensor

def trace_and_check(config_name):
    gc.collect()
    torch.cuda.empty_cache()
    model = build_model(config_name)
    img_tensor = build_input(model)

    # init run
    # some models need to be run 2 times before doing trace
    model.forward_dummy(torch.rand_like(img_tensor))
    model.forward_dummy(torch.rand_like(img_tensor))

    seed = 87432
    with torch.no_grad():
        while True:
            torch.manual_seed(seed)
            out_orig_1 = model.forward_dummy(img_tensor)
            torch.manual_seed(seed)
            out_orig_2 = model.forward_dummy(img_tensor)
            if check_nan(out_orig_1) or check_nan(out_orig_2):
                print('nan found, re-run')
                increment = torch.rand_like(img_tensor)
                img_tensor += increment
            else:
                break

        assert check_equal(out_orig_1, out_orig_2), 'check_equal failure for original model'

    if config_name == 'pvt/retinanet_pvt-l_fpn_1x_coco':
        # to support numpy.intc
        import torch.fx as torch_fx
        from numpy import int64, intc
        orig_base_types = torch_fx.proxy.base_types
        torch_fx.proxy.base_types = (*torch_fx.proxy.base_types, intc, int64)

    with torch.no_grad():
        out_orig = model.forward_dummy(img_tensor)

    traced_gm = concrete_trace(
        model,
        {"img": img_tensor},
        use_operator_patch=False,
        forwrad_function_name="forward_dummy",
        autowrap_leaf_function={
            all: ((), False, None),
            min: ((), False, None),
            max: ((), False, None),
        },
        autowrap_leaf_class={
            int: ((), False),
            reversed: ((), False),
            torch.Size: ((), False),
            type(out_orig): ((), False),
        },
        leaf_module=(
            mmcv_cnn.bricks.wrappers.Conv2d,
            mmcv_cnn.bricks.wrappers.Conv3d,
            mmcv_cnn.bricks.wrappers.ConvTranspose2d,
            mmcv_cnn.bricks.wrappers.ConvTranspose3d,
            mmcv_cnn.bricks.wrappers.Linear,
            mmcv_cnn.bricks.wrappers.MaxPool2d,
            mmcv_cnn.bricks.wrappers.MaxPool3d,
        ),
        fake_middle_class=(mmdet_core.anchor.anchor_generator.AnchorGenerator,),
        dce=True
    )

    if config_name == 'pvt/retinanet_pvt-l_fpn_1x_coco':
        torch_fx.proxy.base_types = orig_base_types

    # seed = torch.seed()
    # with torch.no_grad():
    #     input_like = torch.rand_like(img_tensor)
    #     torch.manual_seed(seed)
    #     out_like = model.forward_dummy(input_like)
    #     torch.manual_seed(seed)
    #     out_like_traced = traced_gm(input_like)
    #     assert check_equal(out_like, out_like_traced), 'check_equal failure in new inputs'

    return traced_gm, model, img_tensor