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

sys.path.append("/home/yileiyang/workspace/DNNGen/concrete_trace_test/nni")
from nni.common.concrete_trace_utils import concrete_trace
from nni.common.concrete_trace_utils.passes import KwargsShapeProp
from nni.common.concrete_trace_utils.passes.counter import counter_pass

config_files_correct = [
    'atss/atss_r50_fpn_1x_coco',
    'autoassign/autoassign_r50_fpn_8x2_1x_coco',
    'cascade_rcnn/cascade_mask_rcnn_r50_caffe_fpn_1x_coco',
    'centernet/centernet_resnet18_140e_coco',
    'centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco',
    'cityscapes/faster_rcnn_r50_fpn_1x_cityscapes',
    'cornernet/cornernet_hourglass104_mstest_8x6_210e_coco',
    'dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco',
    'dcnv2/faster_rcnn_r50_fpn_mdconv_c3-c5_1x_coco',
    'ddod/ddod_r50_fpn_1x_coco',
    'deepfashion/mask_rcnn_r50_fpn_15e_deepfashion',
    'deformable_detr/deformable_detr_r50_16x2_50e_coco',
    'detr/detr_r50_8x2_150e_coco',
    'double_heads/dh_faster_rcnn_r50_fpn_1x_coco',
    'dyhead/atss_r50_caffe_fpn_dyhead_1x_coco',
    'dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco',
    'empirical_attention/faster_rcnn_r50_fpn_attention_0010_1x_coco',
    'faster_rcnn/faster_rcnn_r50_fpn_1x_coco',
    'fcos/fcos_center_r50_caffe_fpn_gn-head_1x_coco',
    'foveabox/fovea_align_r50_fpn_gn-head_4x4_2x_coco',
    'fpg/faster_rcnn_r50_fpg-chn128_crop640_50e_coco',
    'free_anchor/retinanet_free_anchor_r50_fpn_1x_coco',
    'fsaf/fsaf_r50_fpn_1x_coco',
    'gfl/gfl_r50_fpn_1x_coco',
    'ghm/retinanet_ghm_r50_fpn_1x_coco',
    'gn/mask_rcnn_r50_fpn_gn-all_2x_coco',
    'gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco',
    'grid_rcnn/grid_rcnn_r50_fpn_gn-head_1x_coco',
    'groie/faster_rcnn_r50_fpn_groie_1x_coco',
    'hrnet/cascade_mask_rcnn_hrnetv2p_w18_20e_coco',
    'htc/htc_r50_fpn_1x_coco',
    'instaboost/cascade_mask_rcnn_r50_fpn_instaboost_4x_coco',
    'legacy_1.x/faster_rcnn_r50_fpn_1x_coco_v1',
    'lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1',
    'ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco',
    'nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco',
    'nas_fpn/retinanet_r50_fpn_crop640_50e_coco',
    'openimages/faster_rcnn_r50_fpn_32x2_1x_openimages',
    'paa/paa_r50_fpn_1x_coco',
    'pafpn/faster_rcnn_r50_pafpn_1x_coco',
    'pisa/pisa_faster_rcnn_r50_fpn_1x_coco',
    'point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco',
    'pvt/retinanet_pvt-l_fpn_1x_coco',
    'queryinst/queryinst_r50_fpn_1x_coco',
    'regnet/cascade_mask_rcnn_regnetx-400MF_fpn_mstrain_3x_coco',
    'reppoints/bbox_r50_grid_center_fpn_gn-neck+head_1x_coco',
    'res2net/cascade_mask_rcnn_r2_101_fpn_20e_coco',
    'resnet_strikes_back/cascade_mask_rcnn_r50_fpn_rsb-pretrain_1x_coco',
    'retinanet/retinanet_r18_fpn_1x_coco',
    'rpn/rpn_r50_caffe_c4_1x_coco',
    'sabl/sabl_cascade_rcnn_r50_fpn_1x_coco',
    'scratch/faster_rcnn_r50_fpn_gn-all_scratch_6x_coco',
    'sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco',
    'ssd/ssdlite_mobilenetv2_scratch_600e_coco',
    'swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco',
    'tood/tood_r50_fpn_1x_coco',
    'vfnet/vfnet_r2_101_fpn_mdconv_c3-c5_mstrain_2x_coco',
    'yolact/yolact_r50_1x8_coco',
    'yolo/yolov3_d53_320_273e_coco',
    'yolof/yolof_r50_c5_8x8_1x_coco',
    'yolox/yolox_nano_8x8_300e_coco',
]

config_files_need_gpu = [
    'carafe/faster_rcnn_r50_fpn_carafe_1x_coco',
    'efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco',
    'gcnet/cascade_mask_rcnn_x101_32x4d_fpn_syncbn-backbone_1x_coco',
    'resnest/cascade_mask_rcnn_s50_fpn_syncbn-backbone+head_mstrain_1x_coco',
    'selfsup_pretrain/mask_rcnn_r50_fpn_mocov2-pretrain_1x_coco',
]

config_files_no_forward_dummy = [
    'panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco',
    'solo/decoupled_solo_light_r50_fpn_3x_coco',
    'solov2/solov2_light_r18_fpn_3x_coco',
]

# cannot get model:
# 'MaskRCNN: StandardRoIHead: Shared4Conv1FCBBoxHead: Default process group has not been initialized, please make sure to call init_process_group.'
config_files_maskrcnn = [
    'simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco',
    'simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco',
    'strong_baselines/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco',
    'strong_baselines/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_lsj_100e_coco',
]

config_files_img_metas = [
    'nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco',
    'mask2former/mask2former_r50_lsj_8x2_50e_coco',
    'maskformer/maskformer_r50_mstrain_16x1_75e_coco',
    'mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco',
]

# cannot build input: 'proposals'
config_files_proposals = [
    'fast_rcnn/fast_rcnn_r50_caffe_fpn_1x_coco',
    'fast_rcnn/fast_rcnn_r50_fpn_2x_coco',
    'fast_rcnn/fast_rcnn_r101_fpn_2x_coco',
    'fast_rcnn/fast_rcnn_r50_fpn_1x_coco',
    'fast_rcnn/fast_rcnn_r101_caffe_fpn_1x_coco',
    'fast_rcnn/fast_rcnn_r101_fpn_1x_coco',
    'cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco',
    'guided_anchoring/ga_fast_r50_caffe_fpn_1x_coco',
    'libra_rcnn/libra_fast_rcnn_r50_fpn_1x_coco',
]

config_files_other = [
    # cannot compare result
    'lad/lad_r50_paa_r101_fpn_coco_1x',
    # cannot get model: other files do not exist
    'ld/ld_r18_gflv1_r101_fpn_coco_1x',
    'ld/ld_r101_gflv1_r101dcn_fpn_coco_2x',
    # cannot run forward_dummy
    'scnet/scnet_r50_fpn_1x_coco',
    # bad result: output is tensor(nan, nan...), so cannot compare.
    'detectors/cascade_rcnn_r50_rfp_1x_coco',
    # torch.cuda.OutOfMemoryError: CUDA out of memory.
    'dyhead/atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco',
    'faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco',
    'faster_rcnn/faster_rcnn_r50_caffe_c4_mstrain_1x_coco',
    'mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco',
    'pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712',
    'seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1',
    # AssertionError: check_equal failure for original model
    'pisa/pisa_ssd512_coco',
    'ssd/ssd512_coco',
    'tridentnet/tridentnet_r50_caffe_mstrain_3x_coco',
    'pascal_voc/ssd300_voc0712',
    'detectors/detectors_htc_r101_20e_coco',
    'tridentnet/tridentnet_r50_caffe_1x_coco',
    'wider_face/ssd300_wider_face',
    # ImportError
    'timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco',
    'convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco',
    # KeyError: "model"
    'common/mstrain-poly_3x_coco_instance',
    # TypeError
    'seesaw_loss/mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1',
    'objects365/retinanet_r50_fpn_syncbn_1350k_obj365v1',
    # NotImplementedError
    'cascade_rpn/crpn_r50_caffe_fpn_1x_coco',
    'pvt/retinanet_pvt-m_fpn_1x_coco',
    'pvt/retinanet_pvt-s_fpn_1x_coco',
    'solo/solo_r50_fpn_3x_coco',
    'timm_example/retinanet_timm_tv_resnet50_fpn_1x_coco',
    'rfnext/rfnext_fixed_multi_branch_panoptic_fpn_r2_50_fpn_fp16_1x_coco',
    'rfnext/rfnext_search_panoptic_fpn_r2_50_fpn_fp16_1x_coco',
    'rfnext/rfnext_fixed_single_branch_panoptic_fpn_r2_50_fpn_fp16_1x_coco',
    'maskformer/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco',
]

config_files_not_support = [
    *config_files_no_forward_dummy, 
    *config_files_maskrcnn,
    *config_files_img_metas,
    *config_files_proposals,
    *config_files_other
]

exception_dir = [
    "fast_rcnn",
    "panoptic_fpn",
    "common",
    "ld",
    "seesaw_loss",
    "detectors",
    "strong_baselines",
    "solov2",
    "simple_copy_paste",
    "cascade_rpn",
    "pvt",
    "mask2former",
    "solo",
    "scnet",
]


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
if device.type == 'cuda':
    config_files_correct = (*config_files_correct, *config_files_need_gpu)

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
    model = build_model(config_name)
    img_tensor = build_input(model)

    # init run
    # some models need to be run 2 times before doing trace
    model.forward_dummy(torch.rand_like(img_tensor))
    model.forward_dummy(torch.rand_like(img_tensor))

    with torch.no_grad():
        while True:
            seed = torch.seed()
            torch.manual_seed(seed)
            out_orig_1 = model.forward_dummy(img_tensor)
            torch.manual_seed(seed)
            out_orig_2 = model.forward_dummy(img_tensor)
            if check_nan(out_orig_1) or check_nan(out_orig_2):
                print('nan found, re-run')
                img_tensor = build_input(model)
            else:
                break
        try:
            assert check_equal(out_orig_1, out_orig_2), 'check_equal failure for original model'
        except AssertionError:
            with open('out_orig_1', 'w') as f:
                print(out_orig_1, file=f)
            with open('out_orig_2', 'w') as f:
                print(out_orig_2, file=f)
            raise

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

    seed = torch.seed()
    with torch.no_grad():
        input_like = torch.rand_like(img_tensor)
        torch.manual_seed(seed)
        out_like = model.forward_dummy(input_like)
        torch.manual_seed(seed)
        out_like_traced = traced_gm(input_like)
        assert check_equal(out_like, out_like_traced), 'check_equal failure in new inputs'

    return traced_gm, img_tensor

if __name__ == '__main__':
    configs_dir = "/home/yileiyang/workspace/DNNGen/concrete_trace_test/mmdetection/configs"
    
    with open("traced_list", "r") as f:
        traced_list = eval(f.readline())
    with open("failed_list", "r") as f:
        failed_list = eval(f.readline())
    
    traced_count = 0
    total_count = 0
    dir_count = 0
    for dir in os.listdir(configs_dir):
        if dir == "_base_":
            continue
        elif dir in exception_dir:
            configs = os.listdir(os.path.join(configs_dir, dir))
            configs = [config for config in configs if config.endswith(".py")]
            total_count += len(configs)
            continue

        for config_file in os.listdir(os.path.join(configs_dir, dir)):
            if not config_file.endswith(".py"):
                continue
            config_name = dir + "/" + config_file[:-3]
            total_count += 1

            if config_name in traced_list:
                print(f'{config_name} OK')
                traced_count += 1
                continue
            elif config_name in failed_list:
                print(f'{config_name} failed')
                continue
            elif config_name in config_files_not_support:
                continue
            else:
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    traced_gm, img_tensor = trace_and_check(config_name)
                except Exception as e:
                    print(f'{config_name} trace failed')
                    failed_list.append(config_name)
                    with open('exceptions/trace', 'r') as f:
                        trace_exceptions = eval(f.readline())
                    with open('exceptions/trace', 'w') as f:
                        trace_exceptions.append(config_name)
                        print(trace_exceptions, file=f)
                    raise

                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    KwargsShapeProp(traced_gm).propagate({'img': img_tensor})
                except Exception as e:
                    print(f'{config_name} shape_prop failed')
                    failed_list.append(config_name)
                    with open('exceptions/shape_prop', 'r') as f:
                        prop_exceptions = eval(f.readline())
                    with open('exceptions/shape_prop', 'w') as f:
                        prop_exceptions.append(config_name)
                        print(prop_exceptions, file=f)
                    raise

                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    counter_pass(traced_gm, (img_tensor,), verbose=False)
                except Exception as e:
                    print(f'{config_name} count failed')
                    failed_list.append(config_name)
                    with open('exceptions/count', 'r') as f:
                        count_exceptions = eval(f.readline())
                    with open('exceptions/count', 'w') as f:
                        count_exceptions.append(config_name)
                        print(count_exceptions, file=f)
                    raise
                
                traced_count += 1
                traced_list.append(config_name)
                print(f'{config_name} OK')

            if total_count % 5 == 0:
                with open('traced_list', 'w') as f:
                    print(traced_list, file=f)
                with open('failed_list', 'w') as f:
                    print(failed_list, file=f)
                print(f"{traced_count} / {total_count} models traced")

            gc.collect()
            torch.cuda.empty_cache()

    with open('traced_list', 'w') as f:
        print(traced_list, file=f)
    with open('failed_list', 'w') as f:
        print(failed_list, file=f)
    print(f"{traced_count} / {total_count} models traced")
