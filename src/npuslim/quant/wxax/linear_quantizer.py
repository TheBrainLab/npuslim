# Copyright Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from msmodelslim.pytorch.llm_ptq.llm_ptq_tools.quant_funcs import (
    StatMinMaxObserver,
    HistogramObserver,
    fake_quantize,
    linear_quantization_params,
    init_weight_quant_normal,
    init_weight_quant_hessian,
)

_OFFLOAD_DEVICE = "cpu"


class Quantizer(nn.Module):
    """ Quantizer for quantize the tensor"""

    def __init__(self,
                 bit=8,
                 is_signed=True,
                 is_enable=False,
                 is_input=False,
                 cfg=None,
                 logger=None,
                 is_dynamic=False):
        super(Quantizer, self).__init__()
        self.logger = logger
        self.cfg = cfg
        self.bit = bit
        self.is_signed = is_signed
        self.is_enable = is_enable
        self.is_enable_input = is_enable
        self.is_input = is_input
        self.is_sym = cfg.a_sym if is_input else cfg.w_sym
        self.is_calib = True
        self.pr = cfg.pr  # qdrop
        self.mm_per_tensor = cfg.mm_tensor
        self.int_bias = cfg.int_bias
        self.w_hessian = cfg.w_hessian
        self.use_hqq = cfg.hqq
        self.is_dynamic = is_dynamic
        self.int_infer = False
        self.int_weight_flag = False

        # keep accuracy, [bool, int] or bool
        self.admm = cfg.keep_acc['admm']
        self.round_opt = cfg.keep_acc['round_opt']

        self.observer = None

        self.input_scale = None
        self.input_offset = None
        self.weight_scale = None
        self.weight_offset = None

        self.name = None
        self.range_param = None
        self.has_zero = True
        self.quant_weight_tensor = None
        self.int_weight_tensor = None
        self.threshold_disable = 100

        self.register_buffer('x_max', torch.tensor(1.0))
        self.register_buffer('x_min', torch.tensor(1.0))
        self.has_init_quant_para = False

        self.print_flag = True
        self.hessian_optim = cfg.hessian_optim

    def init_act_and_observer(self, cfg):
        if cfg.act_method == 1:
            self.logger.info("use min-max observer:%r, range_parm:%r", self.name, self.range_param)
            self.observer = StatMinMaxObserver(self.bit, self.is_signed, self.is_sym)
        elif cfg.act_method == 2:
            self.logger.info("use histogram observer:%r, range_parm:%r", self.name, self.range_param)
            self.observer = HistogramObserver(qscheme=torch.per_tensor_affine)
        elif cfg.act_method == 3:
            if self.range_param <= 50:
                self.logger.info("use histogram observer:%r, range_parm:%r", self.name, self.range_param)
                self.observer = HistogramObserver(qscheme=torch.per_tensor_affine)
            elif 50 < self.range_param:
                self.logger.info("use min-max observer:%r, range_parm:%r", self.name, self.range_param)
                self.observer = StatMinMaxObserver(self.bit, self.is_signed, self.is_sym)

    def disable_input_quantization(self):
        self.is_enable_input = False

    def enable_quantization(self, name, range_param):
        self.name = name
        self.range_param = range_param
        self.is_enable = True

    def disable_quantization(self, name):
        self.name = name
        self.is_enable = False

    def disable_calib(self):
        self.is_calib = False
        if self.observer is not None:
            self.observer.reset_observer()

    def enable_calib(self):
        self.is_calib = True

    def enable_int_infer(self):
        self.int_infer = True
        self.int_weight_flag = True
        if not self.is_calib:
            # clear the stored quant weight
            self.quant_weight_tensor = None

    def disable_int_infer(self):
        self.int_infer = False
        self.int_weight_flag = False
        self.int_weight_tensor = None

    def get_scale_offset(self):
        if self.is_input:
            return self.input_scale, self.input_offset
        else:
            return self.weight_scale, self.weight_offset

    def set_ratio(self, ratio=0.9):
        if self.observer is not None:
            self.observer.set_ratio(ratio)

    def new_quant_tensor(self, data):
        _, data_dequant = fake_quantize(
            data, self.input_scale, self.input_offset, self.bit, is_signed=self.is_signed,
        )

        if self.is_calib:
            prob = self.pr
            if self.print_flag:
                self.logger.info(
                    "layer: %r, range: %r, Automatically set the drop rate: %r", self.name, self.range_param, prob
                )
                self.print_flag = False

            if (prob < 1.0) and (prob >= 0.0):
                # element-wise q-drop
                x_dequant = torch.where(torch.rand_like(data) < prob, data_dequant, data)
            elif prob < 0.0:
                p = np.random.rand()
                p = 0.0 if p < 0.5 else p
                # layer-wise or element-wise q-drop, depending on p
                x_dequant = torch.where(torch.rand_like(data) < p, data_dequant, data)
            else:
                # no q-drop
                x_dequant = data_dequant
        else:
            # no q-drop, inference with fake quantization
            x_dequant = data_dequant

        return x_dequant

    def tensor_forward(self, tensor, y=None):
        if not self.is_enable:
            return tensor
        if self.is_input and not self.is_enable_input:
            return tensor

        # value check of forward tensor
        if tensor.numel() == 0:
            self.disable_quantization(self.name)
            return tensor

        # weight quantization
        with torch.no_grad():
            if not self.is_input:
                return self._quant_weight_forward(tensor, y)

            # activation quantization
            if self.is_dynamic:
                self._stat_dynamic_input(tensor)
            return self._quant_activation_forward(tensor)

    def get_anti_outlier(self, k, current_t):
        current_std = np.std(current_t)
        current_mean = np.mean(current_t)
        threshold1 = current_mean - k * current_std
        threshold2 = current_mean + k * current_std
        bigger_num = np.sum(current_t >= threshold2)
        smaller_num = np.sum(current_t <= threshold1)
        res = (bigger_num + smaller_num) / current_t.size
        return res

    def _init_weight_quant_normal(self, weight, y, integral_zero_point=True):
        if self.has_init_quant_para:
            return
        calling_params = self.bit, self.is_sym, self.is_signed, integral_zero_point, self.admm
        if self.w_hessian:
            if self.hessian_optim['flag']:
                current_t = weight.data.cpu().numpy().reshape(1, -1)[0]
                threshold = self.get_anti_outlier(self.sigma_weight, current_t)
                if threshold > self.hessian_optim['std_threshold']:
                    # weight optimization based on Hessian information
                    _, _, self.weight_scale, self.weight_offset = \
                        init_weight_quant_hessian(weight, y, *calling_params, mm_tensor=self.mm_per_tensor)
                else:
                    self.w_hessian = False
                    _, _, self.weight_scale, self.weight_offset = \
                        init_weight_quant_normal(
                            weight, *calling_params, mm_tensor=self.mm_per_tensor, hqq=self.use_hqq
                        )
            else:
                # weight optimization based on Hessian information
                _, _, self.weight_scale, self.weight_offset = \
                    init_weight_quant_hessian(weight, y, *calling_params, self.mm_per_tensor)
        else:
            _, _, self.weight_scale, self.weight_offset = \
                init_weight_quant_normal(
                    weight,
                    *calling_params,
                    round_opt=self.round_opt,
                    mm_tensor=self.mm_per_tensor,
                    hqq=self.use_hqq
                )
        self.has_init_quant_para = True

    def _stat_dynamic_input(self, tensor):
        """dynamic场景下获取当前input的tensor的min和max."""
        if tensor.dim() == 2:
            tensor_tmp = tensor.unsqueeze(0)
            x_min = tensor_tmp.min(2)[0]
            x_max = tensor_tmp.max(2)[0]
        else:
            x_min = tensor.min(2)[0]
            x_max = tensor.max(2)[0]

        self.x_min = x_min.view(-1, x_min.shape[1], 1)
        self.x_max = x_max.view(-1, x_max.shape[1], 1)

    def _quant_weight_forward(self, tensor, y):
        self._init_weight_quant_normal(tensor, y)
        if (not self.is_calib) and self.int_infer:
            if self.int_weight_flag:
                self.int_weight_tensor, _ = fake_quantize(
                    tensor=tensor,
                    scale=self.weight_scale,
                    zero_point=self.weight_offset,
                    bit=self.bit,
                    is_signed=self.is_signed,
                    dequant=False
                )
                self.int_weight_flag = False
            return self.int_weight_tensor
        else:
            _, quant_weight_tensor = fake_quantize(
                tensor=tensor,
                scale=self.weight_scale,
                zero_point=self.weight_offset,
                bit=self.bit,
                is_signed=self.is_signed,
                dequant=True
            )
            return quant_weight_tensor

    def _quant_activation_forward(self, tensor):
        if self.is_dynamic:
            self.input_scale, self.input_offset = linear_quantization_params(
                self.bit, self.x_min, self.x_max, q_signed=self.is_signed, sym=self.is_sym
            )

        dtype = tensor.dtype
        if (not self.is_calib) and self.int_infer:
            # int8*int8, offset correction
            if dtype == torch.bfloat16:
                self.input_scale = self.input_scale.to(dtype)
                self.input_offset = self.input_offset.to(dtype)
            int_tensor, _ = fake_quantize(
                tensor=tensor,
                scale=self.input_scale,
                zero_point=self.input_offset,
                bit=self.bit,
                is_signed=self.is_signed,
                dequant=False
            )
            return int_tensor

        if self.is_calib:
            self.observer.update(tensor)
            self.x_min, self.x_max = self.observer.get_min_max(self.x_min.device)
            self.input_scale, self.input_offset = linear_quantization_params(
                bit=self.bit,
                x_min=self.x_min,
                x_max=self.x_max,
                q_signed=self.is_signed,
                sym=self.is_sym
            )
            if dtype == torch.bfloat16:
                self.input_scale = self.input_scale.to(dtype)
                self.input_offset = self.input_offset.to(dtype)
        return self.new_quant_tensor(tensor)


class TensorQuantizer(Quantizer):
    """
    Class to quantize given tensor
    """

    def __init__(self, **kwargs):
        super(TensorQuantizer, self).__init__(**kwargs)

    def forward(self, tensor, y=None):
        return self.tensor_forward(tensor, y)


class LinearQuantizer(nn.Module):
    """
    Class to quantize given linear layer weights
    """

    def __init__(self, cfg=None, logger=None):
        """
        cfg: quantizaton configuration
        """
        super(LinearQuantizer, self).__init__()
        self.in_features = None
        self.out_features = None
        self.weight = None
        self.bias = None
        self.quant_input = TensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg, logger=logger, is_dynamic=cfg.is_dynamic
        )
        self.quant_weight = TensorQuantizer(
            bit=cfg.w_bit, is_signed=cfg.w_signed, is_enable=True,
            is_input=False, cfg=cfg, logger=logger
        )
        self.cfg = cfg

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data)
        try:
            self.bias = nn.Parameter(linear.bias.data)
        except AttributeError:
            self.bias = None
        finally:
            pass

    def set_ratio(self, ratio=0.9):
        self.quant_input.set_ratio(ratio)

    def activation_calib(self, x):
        self.quant_input(x)

    def weight_calib(self, y=None):
        self.quant_weight(self.weight, y)

    def forward(self, x):
        if self.quant_weight.int_infer and (not self.quant_weight.is_calib):
            return self._int_infer_forward(x)
        else:
            if self.quant_input.w_hessian:  # gptq
                weight = self.quant_weight(self.weight, y=x.clone())
            else:
                weight = self.quant_weight(self.weight)
            if self.quant_input.bit <= 8:
                x = self.quant_input(x)
            return F.linear(x, weight, self.bias)

    def _int_infer_forward(self, x):
        ori_dtype = x.dtype
        quant_param_dtype = torch.float32
        x_dtype = torch.float16
        if ori_dtype == torch.bfloat16:
            quant_param_dtype = torch.bfloat16
            x_dtype = torch.bfloat16

        x = x.to(x_dtype)
        x = self.quant_input(x)
        weight = self.quant_weight(self.weight)
        x = x.to(quant_param_dtype)
        weight = weight.to(quant_param_dtype)
        int_out = F.linear(x, weight)
        input_scale, input_offset = self.quant_input.get_scale_offset()
        weight_scale, _ = self.quant_weight.get_scale_offset()
        if (input_scale is not None) and (weight_scale is not None):
            input_scale = input_scale.to(quant_param_dtype)
            input_offset = input_offset.to(quant_param_dtype)
            weight_scale = weight_scale.to(quant_param_dtype)
            if len(weight_scale.shape) > 1:
                weight_scale = weight_scale.reshape((len(x.shape) - 2) * (1,) + (1, weight.shape[0]))
            # offset correction, offline calibration
            correction = weight.sum(dim=1) * input_offset.to(quant_param_dtype)
            fp_scale = input_scale * weight_scale
            fp_scale = fp_scale.to(quant_param_dtype)
            fp_out = self._bias_and_dequant_process(correction, int_out, fp_scale, x.device, x_dtype)
        else:
            fp_out = int_out
            if self.bias is not None:
                fp_out += self.bias.data

        return fp_out.to(ori_dtype)

    def _bias_and_dequant_process(self, correction, int_out, fp_scale, x_device, x_dtype):
        if self.quant_weight.int_bias:
            # int32 bias, offline calibration
            if self.bias is not None:
                bias_int = (self.bias.data / fp_scale).round()
            else:
                bias_int = torch.zeros(correction.size(0)).to(x_device)
            bias_int -= correction

            # int32 biasadd -> dequant
            fp_out = (int_out + bias_int) * fp_scale
            fp_out = fp_out.to(x_dtype)
        else:
            # dequant
            fp_out = int_out * fp_scale
            fp_out = fp_out.to(x_dtype)

            # fp32 bias, offline calibration
            correction_fp = correction * fp_scale
            correction_fp = correction_fp.to(x_dtype)
            if self.bias is not None:
                bias_fp = self.bias.data.to(x_dtype) - correction_fp
            else:
                bias_fp = - correction_fp
            # fp32 biasadd
            fp_out += bias_fp
        return fp_out


class Conv2dQuantizer(nn.Module):
    """
    Class to quantize given convolutional layer weights
    """

    def __init__(self, cfg=None, logger=None):
        """
        cfg: quantizaton configuration
        """
        super(Conv2dQuantizer, self).__init__()
        self.in_channels = None
        self.out_channels = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.dilation = None
        self.groups = None
        self.weight = None
        self.bias = None
        self.quant_input = TensorQuantizer(
            bit=cfg.a_bit, is_signed=cfg.a_signed, is_enable=True,
            is_input=True, cfg=cfg, logger=logger
        )
        self.quant_weight = TensorQuantizer(
            bit=cfg.w_bit, is_signed=cfg.w_signed, is_enable=True,
            cfg=cfg, logger=logger
        )

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.data)
        try:
            self.bias = nn.Parameter(conv.bias.data)
        except AttributeError:
            self.bias = None
        finally:
            pass

    def forward(self, x):
        x = self.quant_input(x)
        weight = self.quant_weight(self.weight)
        return self._conv_forward(x, weight)

    def _conv_forward(self, x, weight):
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )