import paddle.signal as signal
import paddle.incubate.nn.functional as F
from pathlib import Path
import inspect
import torch

test_sig_list = []
# Define the list of methods to test for signatures
no_signature_methods = []

# Combine all methods to test
all_methods = test_sig_list + [
    "paddle.add",
    "paddle.as_strided",
    "paddle.atleast_1d",
    "paddle.atleast_2d",
    "paddle.atleast_3d",
    "paddle.audio.functional.get_window",
    "paddle.bitwise_invert",
    "paddle.broadcast_shape",
    "paddle.broadcast_tensors",
    "paddle.crop",
    "paddle.cummax",
    "paddle.cummin",
    "paddle.cumprod",
    "paddle.empty_like",
    "paddle.equal_all",
    "paddle.floor_mod",
    "paddle.frexp",
    "paddle.gammaln",
    "paddle.greater_equal",
    "paddle.greater_than",
    "paddle.histogram",
    "paddle.histogram_bin_edges",
    "paddle.histogramdd",
    "paddle.increment",
    "paddle.incubate.nn.functional.blha_get_max_len",
    "paddle.incubate.nn.functional.fused_bias_act",
    "paddle.incubate.nn.functional.fused_bias_dropout_residual_layer_norm",
    "paddle.incubate.nn.functional.fused_dropout_add",
    "paddle.incubate.nn.functional.fused_feedforward",
    "paddle.incubate.nn.functional.fused_layer_norm",
    "paddle.incubate.nn.functional.fused_linear",
    "paddle.incubate.nn.functional.fused_linear_activation",
    "paddle.incubate.nn.functional.fused_matmul_bias",
    "paddle.incubate.softmax_mask_fuse",
    "paddle.is_empty",
    "paddle.kthvalue",
    "paddle.lcm",
    "paddle.ldexp",
    "paddle.less",
    "paddle.less_equal",
    "paddle.less_than",
    "paddle.linalg.corrcoef",
    "paddle.linalg.eig",
    "paddle.linalg.eigvals",
    "paddle.linalg.eigvalsh",
    "paddle.linalg.lu",
    "paddle.linalg.matrix_transpose",
    "paddle.linalg.qr",
    "paddle.linalg.slogdet",
    "paddle.linalg.svd",
    "paddle.linalg.triangular_solve",
    "paddle.log_normal",
    "paddle.masked_scatter",
    "paddle.median",
    "paddle.mod",
    "paddle.mode",
    "paddle.nanmedian",
    "paddle.negative",
    "paddle.nn.functional.avg_pool1d",
    "paddle.nn.functional.avg_pool2d",
    "paddle.nn.functional.avg_pool3d",
    "paddle.nn.functional.batch_norm",
    "paddle.nn.functional.bilinear",
    "paddle.nn.functional.channel_shuffle",
    "paddle.nn.functional.class_center_sample",
    "paddle.nn.functional.conv1d_transpose",
    "paddle.nn.functional.conv2d",
    "paddle.nn.functional.conv3d",
    "paddle.nn.functional.conv3d_transpose",
    "paddle.nn.functional.cosine_embedding_loss",
    "paddle.nn.functional.dropout",
    "paddle.nn.functional.dropout2d",
    "paddle.nn.functional.dropout3d",
    "paddle.nn.functional.flashmask_attention",
    "paddle.nn.functional.fractional_max_pool2d",
    "paddle.nn.functional.fractional_max_pool3d",
    "paddle.nn.functional.gelu",
    "paddle.nn.functional.group_norm",
    "paddle.nn.functional.hinge_embedding_loss",
    "paddle.nn.functional.instance_norm",
    "paddle.nn.functional.kl_div",
    "paddle.nn.functional.l1_loss",
    "paddle.nn.functional.label_smooth",
    "paddle.nn.functional.local_response_norm",
    "paddle.nn.functional.log_loss",
    "paddle.nn.functional.log_softmax",
    "paddle.nn.functional.lp_pool1d",
    "paddle.nn.functional.lp_pool2d",
    "paddle.nn.functional.margin_ranking_loss",
    "paddle.nn.functional.max_pool1d",
    "paddle.nn.functional.max_pool2d",
    "paddle.nn.functional.max_pool3d",
    "paddle.nn.functional.maxout",
    "paddle.nn.functional.mse_loss",
    "paddle.nn.functional.multi_label_soft_margin_loss",
    "paddle.nn.functional.npair_loss",
    "paddle.nn.functional.pixel_shuffle",
    "paddle.nn.functional.pixel_unshuffle",
    "paddle.nn.functional.prelu",
    "paddle.nn.functional.scaled_dot_product_attention",
    "paddle.nn.functional.selu",
    "paddle.nn.functional.smooth_l1_loss",
    "paddle.nn.functional.soft_margin_loss",
    "paddle.nn.functional.square_error_cost",
    "paddle.nn.functional.swish",
    "paddle.nn.functional.temporal_shift",
    "paddle.nn.quant.weight_quantize",
    "paddle.nonzero",
    "paddle.not_equal",
    "paddle.numel",
    "paddle.poisson",
    "paddle.polar",
    "paddle.positive",
    "paddle.pow",
    "paddle.rank",
    "paddle.reduce_as",
    "paddle.remainder",
    "paddle.reverse",
    "paddle.round",
    "paddle.scale",
    "paddle.searchsorted",
    "paddle.shape",
    "paddle.signal.stft",
    "paddle.slice_scatter",
    "paddle.sort",
    "paddle.standard_gamma",
    "paddle.stanh",
    "paddle.subtract",
    "paddle.take",
    "paddle.Tensor.__abs__",
    "paddle.Tensor.__add__",
    "paddle.Tensor.__and__",
    "paddle.Tensor.__dir__",
    "paddle.Tensor.__div__",
    "paddle.Tensor.__eq__",
    "paddle.Tensor.__floordiv__",
    "paddle.Tensor.__ge__",
    "paddle.Tensor.__gt__",
    "paddle.Tensor.__le__",
    "paddle.Tensor.__len__",
    "paddle.Tensor.__lshift__",
    "paddle.Tensor.__lt__",
    "paddle.Tensor.__matmul__",
    "paddle.Tensor.__mod__",
    "paddle.Tensor.__ne__",
    "paddle.Tensor.__neg__",
    "paddle.Tensor.__nonzero__",
    "paddle.Tensor.__or__",
    "paddle.Tensor.__pow__",
    "paddle.Tensor.__radd__",
    "paddle.Tensor.__rlshift__",
    "paddle.Tensor.__rmatmul__",
    "paddle.Tensor.__rmod__",
    "paddle.Tensor.__rmul__",
    "paddle.Tensor.__ror__",
    "paddle.Tensor.__rpow__",
    "paddle.Tensor.__rrshift__",
    "paddle.Tensor.__rshift__",
    "paddle.Tensor.__rsub__",
    "paddle.Tensor.__rtruediv__",
    "paddle.Tensor.__rxor__",
    "paddle.Tensor.__truediv__",
    "paddle.Tensor.__xor__",
    "paddle.Tensor.abs",
    "paddle.Tensor.add",
    "paddle.Tensor.all",
    "paddle.Tensor.any",
    "paddle.Tensor.astype",
    "paddle.Tensor.atanh",
    "paddle.Tensor.bernoulli_",
    "paddle.Tensor.cast",
    "paddle.Tensor.cauchy_",
    "paddle.Tensor.ceil",
    "paddle.Tensor.conj",
    "paddle.Tensor.cos",
    "paddle.Tensor.cumprod",
    "paddle.Tensor.detach",
    "paddle.Tensor.digamma",
    "paddle.Tensor.dim",
    "paddle.Tensor.divide",
    "paddle.Tensor.equal_all",
    "paddle.Tensor.erfinv",
    "paddle.Tensor.exp",
    "paddle.Tensor.exponential_",
    "paddle.Tensor.fill_diagonal_tensor",
    "paddle.Tensor.floor",
    "paddle.Tensor.frexp",
    "paddle.Tensor.geometric_",
    "paddle.Tensor.greater_equal",
    "paddle.Tensor.imag",
    "paddle.Tensor.inverse",
    "paddle.Tensor.is_complex",
    "paddle.Tensor.isinf",
    "paddle.Tensor.isnan",
    "paddle.Tensor.item",
    "paddle.Tensor.less",
    "paddle.Tensor.lgamma",
    "paddle.Tensor.log",
    "paddle.Tensor.log10",
    "paddle.Tensor.log1p",
    "paddle.Tensor.logical_not",
    "paddle.Tensor.max",
    "paddle.Tensor.min",
    "paddle.Tensor.mod",
    "paddle.Tensor.multiply",
    "paddle.Tensor.neg",
    "paddle.Tensor.nonzero",
    "paddle.Tensor.norm",
    "paddle.Tensor.normal_",
    "paddle.Tensor.not_equal",
    "paddle.Tensor.pow",
    "paddle.Tensor.prod",
    "paddle.Tensor.rad2deg",
    "paddle.Tensor.rank",
    "paddle.Tensor.real",
    "paddle.Tensor.reciprocal",
    "paddle.Tensor.round",
    "paddle.Tensor.rsqrt",
    "paddle.Tensor.scale",
    "paddle.Tensor.set_",
    "paddle.Tensor.sigmoid",
    "paddle.Tensor.sign",
    "paddle.Tensor.sin",
    "paddle.Tensor.slice",
    "paddle.Tensor.slice_scatter",
    "paddle.Tensor.sort",
    "paddle.Tensor.split",
    "paddle.Tensor.sqrt",
    "paddle.Tensor.square",
    "paddle.Tensor.subtract",
    "paddle.Tensor.tanh",
    "paddle.Tensor.tolist",
    "paddle.Tensor.transpose",
    "paddle.Tensor.trunc",
    "paddle.Tensor.zero_",
    "paddle.tolist",
    "paddle.unfold",
    "paddle.unstack",
    "paddle.vander",
    "paddle.vecdot",
    "paddle.view_as",
    "paddle.vision.ops.box_coder",
    "paddle.vision.ops.matrix_nms",
    "paddle.vision.ops.prior_box",
    "paddle.vision.ops.yolo_box",
    "paddle.vision.ops.yolo_loss",
    "paddle.add_n",
    "paddle.allclose",
    "paddle.arange",
    "paddle.assign",
    "paddle.autograd.hessian",
    "paddle.autograd.jacobian",
    "paddle.bincount",
    "paddle.binomial",
    "paddle.cartesian_prod",
    "paddle.cast",
    "paddle.divide",
    "paddle.einsum",
    "paddle.empty",
    "paddle.expand",
    "paddle.expand_as",
    "paddle.floor_divide",
    "paddle.incubate.nn.functional.fused_rms_norm",
    "paddle.incubate.nn.functional.fused_rotary_position_embedding",
    "paddle.incubate.nn.functional.swiglu",
    "paddle.incubate.nn.functional.variable_length_memory_efficient_attention",
    "paddle.incubate.softmax_mask_fuse_upper_triangle",
    "paddle.index_fill",
    "paddle.linalg.eigh",
    "paddle.linalg.svdvals",
    "paddle.logaddexp",
    "paddle.matmul",
    "paddle.matrix_transpose",
    "paddle.max",
    "paddle.meshgrid",
    "paddle.min",
    "paddle.multiply",
    "paddle.nn.functional.adaptive_avg_pool2d",
    "paddle.nn.functional.adaptive_avg_pool3d",
    "paddle.nn.functional.alpha_dropout",
    "paddle.nn.functional.binary_cross_entropy_with_logits",
    "paddle.nn.functional.conv1d",
    "paddle.nn.functional.conv2d_transpose",
    "paddle.nn.functional.ctc_loss",
    "paddle.nn.functional.feature_alpha_dropout",
    "paddle.nn.functional.interpolate",
    "paddle.nn.functional.linear",
    "paddle.nn.functional.max_unpool1d",
    "paddle.nn.functional.max_unpool2d",
    "paddle.nn.functional.max_unpool3d",
    "paddle.nn.functional.multi_margin_loss",
    "paddle.nn.functional.pad",
    "paddle.nn.functional.poisson_nll_loss",
    "paddle.nn.functional.sequence_mask",
    "paddle.nn.functional.sigmoid_focal_loss",
    "paddle.nn.functional.softmax",
    "paddle.nn.functional.zeropad2d",
    "paddle.nn.quant.weight_only_linear",
    "paddle.normal",
    "paddle.ones",
    "paddle.prod",
    "paddle.slice",
    "paddle.split",
    "paddle.standard_normal",
    "paddle.strided_slice",
    "paddle.tanh",
    "paddle.Tensor.__mul__",
    "paddle.Tensor.__sub__",
    "paddle.Tensor.expand",
    "paddle.Tensor.log_normal_",
    "paddle.Tensor.reshape",
    "paddle.Tensor.tile",
    "paddle.topk",
    "paddle.view",
    "paddle.vision.ops.decode_jpeg",
    "paddle.vision.ops.deform_conv2d",
    "paddle.vision.ops.distribute_fpn_proposals",
    "paddle.vision.ops.generate_proposals",
    "paddle.vision.ops.nms",
    "paddle.where",
    "paddle.zeros",
    "paddle.bernoulli",
    "paddle.gammainc",
    "paddle.gammaincc",
    "paddle.gather",
    "paddle.gather_nd",
    "paddle.geometric.sample_neighbors",
    "paddle.geometric.segment_max",
    "paddle.geometric.segment_mean",
    "paddle.geometric.segment_min",
    "paddle.geometric.segment_sum",
    "paddle.geometric.send_u_recv",
    "paddle.geometric.send_ue_recv",
    "paddle.geometric.send_uv",
    "paddle.incubate.nn.functional.block_multihead_attention",
    "paddle.incubate.nn.functional.fused_multi_head_attention",
    "paddle.incubate.nn.functional.masked_multihead_attention",
    "paddle.incubate.segment_max",
    "paddle.incubate.segment_mean",
    "paddle.incubate.segment_min",
    "paddle.incubate.segment_sum",
    "paddle.index_add",
    "paddle.index_put",
    "paddle.index_sample",
    "paddle.index_select",
    "paddle.linalg.cholesky",
    "paddle.linalg.lu_unpack",
    "paddle.linalg.pca_lowrank",
    "paddle.multinomial",
    "paddle.multiplex",
    "paddle.nn.functional.adaptive_log_softmax_with_loss",
    "paddle.nn.functional.binary_cross_entropy",
    "paddle.nn.functional.cross_entropy",
    "paddle.nn.functional.dice_loss",
    "paddle.nn.functional.embedding",
    "paddle.nn.functional.gather_tree",
    "paddle.nn.functional.gaussian_nll_loss",
    "paddle.nn.functional.hsigmoid_loss",
    "paddle.nn.functional.margin_cross_entropy",
    "paddle.nn.functional.nll_loss",
    "paddle.nn.functional.one_hot",
    "paddle.nn.functional.rnnt_loss",
    "paddle.nn.functional.softmax_with_cross_entropy",
    "paddle.nn.functional.sparse_attention",
    "paddle.nn.functional.upsample",
    "paddle.put_along_axis",
    "paddle.scatter",
    "paddle.scatter_nd",
    "paddle.scatter_nd_add",
    "paddle.shard_index",
    "paddle.Tensor.coalesce",
    "paddle.Tensor.gather",
    "paddle.Tensor.gather_nd",
    "paddle.Tensor.index_select",
    "paddle.Tensor.is_coalesced",
    "paddle.Tensor.put_along_axis",
    "paddle.Tensor.unflatten",
    "paddle.vision.ops.psroi_pool",
    "paddle.vision.ops.roi_align",
    "paddle.vision.ops.roi_pool",
]

# Function to safely get signature and handle methods without signatures
def get_signature_safely(func_name):
    try:
        func = eval(func_name)
        signature = inspect.signature(func)
        return signature
    except ValueError:
        # This function doesn't have a signature
        no_signature_methods.append(func_name)
        return None

# Test all methods in the list
for method in all_methods:
    sig = get_signature_safely(method)

print("\nMethods without signatures:")
for method in no_signature_methods:
    print(f"- {method}")