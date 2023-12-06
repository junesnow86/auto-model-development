import warnings

warnings.filterwarnings('ignore')

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
from torch.nn.functional import normalize

sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/nni")
sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/")
from dnngen_utils.fxgraph_to_seq import Sequence, fold_seq
from nni.common.concrete_trace_utils import concrete_trace
from nni.common.concrete_trace_utils.passes import KwargsShapeProp
from nni.common.concrete_trace_utils.passes import counter_pass as my_counter_pass


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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build(config_name):
    mmdet_dir = '/home/yileiyang/workspace/DNNGen/concrete_trace_test/mmdetection'
    img = f'{mmdet_dir}/tests/data/color.jpg'
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

    with torch.no_grad():
        torch.manual_seed(867309)
        config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)
        test_pipeline = Compose(config.data.test.pipeline)
        img_data = test_pipeline(dict(img_info=dict(filename=img), img_prefix=None))['img']
        img_tensor = collate(img_data, 1).data[0].to(device)

    model = init_detector(config, device=device)
    model.eval()
    print(type(model))
    return model, img_tensor

def trace_and_check(model, img_tensor):
    # init run
    # RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet. Caffe2 uses a lazy allocation, so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.
    model.forward_dummy(torch.rand_like(img_tensor))
    model.forward_dummy(torch.rand_like(img_tensor))

    seed = 867309
    with torch.no_grad():
        torch.manual_seed(seed)
        out_orig_1 = model.forward_dummy(img_tensor)
        torch.manual_seed(seed)
        out_orig_2 = model.forward_dummy(img_tensor)
        try:
            assert check_equal(out_orig_1, out_orig_2), 'check_equal failed for original model'
        except AssertionError:
            with open('out_orig_1', 'w') as f:
                print(out_orig_1, file=f)
            with open('out_orig_2', 'w') as f:
                print(out_orig_2, file=f)
            raise
    print('check for original model pass')

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

    with torch.no_grad():
        input_like = torch.rand_like(img_tensor)
        torch.manual_seed(seed)
        out_like = model.forward_dummy(input_like)
        torch.manual_seed(seed)
        out_like_traced = traced_gm(input_like)
        assert check_equal(out_like, out_like_traced), 'check_equal failure in new inputs'

    print('trace pass')
    return traced_gm

if __name__ == '__main__':
    # config_name = 'pafpn/faster_rcnn_r50_pafpn_1x_coco'
    # config_name = 'centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco'
    # config_name = 'cityscapes/mask_rcnn_r50_fpn_1x_cityscapes'
    # config_name = 'ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x_coco'
    # config_name = 'openimages/ssd300_32x8_36e_openimages'
    # config_name = 'pascal_voc/ssd512_voc0712'
    config_name = 'tridentnet/tridentnet_r50_caffe_mstrain_1x_coco'
    # config_name = 'res2net/htc_r2_101_fpn_20e_coco'
    # config_name = 'ssd/ssd300_fp16_coco'
    # config_name = 'pisa/pisa_ssd300_coco'
    model, img_tensor = build(config_name)
    traced_gm = trace_and_check(model, img_tensor)

    # with open('graph', 'w') as f:
    #     print(traced_gm.graph, file=f)
    # with open('code.py', 'w') as f:
    #     print(traced_gm.code, file=f)

    KwargsShapeProp(traced_gm).propagate((img_tensor, ))
    seq = Sequence(traced_gm, model)
    with open(f'xml/{config_name.replace("/", "--")}', 'w') as f:
        print(fold_seq(seq), file=f)
    print('shape prop pass')
    my_counter_pass(traced_gm, (img_tensor, ))
    print('count pass')
    print('All done!')
