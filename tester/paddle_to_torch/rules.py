import re
import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class Code:
    """Paddle2PyTorch 转换代码数据类，封装转换后的可执行代码，自动预编译

    Attributes:
        valid: 是否有效，默认为 True
        error_message: 编译错误信息，仅当 valid = False 时有效

        preprocess: 预处理代码，在核心逻辑前执行
        core: 核心逻辑代码，应包含 Torch API
        postprocess: 后处理代码，在核心逻辑后执行

        preprocess_compiled: 预编译的预处理代码
        core_compiled: 预编译的核心逻辑代码
        postprocess_compiled: 预编译的后处理代码
    """

    valid: bool = True
    error_message: Optional[str] = field(default=None, init=False)

    preprocess: List[str] = field(default_factory=list)
    core: List[str] = field(default_factory=list)
    postprocess: List[str] = field(default_factory=list)

    preprocess_compiled: Optional[types.CodeType] = field(init=False, default=None)
    core_compiled: Optional[types.CodeType] = field(init=False, default=None)
    postprocess_compiled: Optional[types.CodeType] = field(init=False, default=None)

    def __post_init__(self):
        """自动编译代码"""
        try:
            self.preprocess_compiled = self._compile(self.preprocess)
            self.core_compiled = self._compile(self.core)
            self.postprocess_compiled = self._compile(self.postprocess)
        except Exception as e:
            self.preprocess_compiled = None
            self.core_compiled = None
            self.postprocess_compiled = None
            self.valid = False
            self.error_message = str(e)

    @classmethod
    def _compile(cls, code_lines: List[str]) -> Optional[types.CodeType]:
        """代码编译方法"""
        if not code_lines:
            return None
        try:
            return compile("\n".join(code_lines), "<string>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in code: {e.msg}") from e

    def is_valid(self) -> bool:
        """检查代码是否编译成功"""
        return self.valid


@dataclass
class ConvertResult:
    """Paddle2PyTorch 转换结果数据类, 封装 API 转换结果，提供成功/失败的构造方法

    Attributes:
        paddle_api (str): Paddle API 名称
        is_supported (bool): 是否支持转换, 默认为 True
        is_torch_corresponding: 是否与 Torch API 对应，默认为 True
        code (Optional[Code]): 转换后的代码数据对象
        output_var (Optional[str]): 输出变量名，默认值 None 表示 result 保存最后的输出值
        error_message (Optional[str]): 错误信息, 仅当 is_supported = False 时有效

    Methods:
        success(paddle_api, code, output_var): 创建成功转换结果
        error(paddle_api, message): 创建失败转换结果
    """

    paddle_api: str
    is_supported: bool = True
    is_torch_corresponding: bool = True

    code: Optional[Code] = None
    output_var: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def success(
        cls,
        paddle_api: str,
        code: Union[Code, List[str]],
        output_var: str = "result",
        is_torch_corresponding: bool = True,
    ) -> "ConvertResult":
        code_obj = Code(core=code) if isinstance(code, list) else code
        if not code_obj.is_valid():
            return cls.error(paddle_api, f"Invalid code: {code_obj.error_message}")

        if is_torch_corresponding and len(code_obj.core) > 6:
            print(
                f"Warning: The core code of {paddle_api} is too complex.",
                flush=True,
            )

        return cls(
            paddle_api,
            code=code_obj,
            output_var=output_var,
            is_torch_corresponding=is_torch_corresponding,
        )

    @classmethod
    def error(cls, paddle_api: str, message: str) -> "ConvertResult":
        return cls(paddle_api, is_supported=False, error_message=message)


class BaseRule(ABC):
    """转换规则的抽象基类"""

    @abstractmethod
    def apply(self, paddle_api: str) -> ConvertResult:
        """
        将 Paddle API 调用转换为 PyTorch 等效代码形式
        code 中可包含输入变量的占位符(如 {input}、{x}), 这些变量将被自动填充为 torch tensor

        Args:
            paddle_api (str): Paddle API 名称

        Returns:
            ConvertResult: 包含代码和输出变量的 ConvertResult 对象, 或错误信息
        """
        pass

    @classmethod
    def _format_arg(cls, arg) -> str:
        """
        将参数格式化为调用字符串的辅助方法

        Args:
            arg: 待格式化的参数

        Returns:
            str: 格式化后的参数
        """
        PLACEHOLDER_PATTERN: re.Pattern = re.compile(r"\{([^{}]+)\}")

        def replacer(match):
            placeholder = match.group(1)
            if placeholder.isdigit():
                return f"_tmp_{placeholder}"
            elif placeholder.replace("_", "").isalnum():
                return placeholder
            return match.group(0)

        if isinstance(arg, str):
            arg = PLACEHOLDER_PATTERN.sub(replacer, arg)
        return str(arg)

    def read_mapping(self, mapping: Dict):
        """
        预处理，根据传入的 json 配置初始化成员变量

        Args:
            mapping (Dict): 包含 json 配置的字典

        Returns:
            None
        """
        self.mapping: Dict = mapping
        if "Rule" in mapping:
            if "torch_api" in mapping:
                self.torch_api: str = mapping["torch_api"]
            return
        if "torch_api" not in mapping:
            raise ValueError("Missing required field 'torch_api' in the mapping.")
        self.torch_api: str = mapping.get("torch_api", "")
        self.args_map: OrderedDict = mapping.get("paddle_torch_args_map", {})
        self.torch_args: List = mapping.get("torch_args", [])
        self.torch_kwargs: OrderedDict = mapping.get("torch_kwargs", OrderedDict())
        self.is_attribute: bool = mapping.get("is_attribute", False)
        self.defaults: Dict = mapping.get("set_defaults", {})

    def apply_generic(self):
        # if "torch_api" in self.mapping:
        #     self.torch_api: str = self.mapping.get("torch_api", "")
        defaults_code = []
        if "set_defaults" in self.mapping:
            defaults = self.mapping.get("set_defaults", {})
            for default_name, default_value in defaults.items():
                defaults_code.append(
                    f"{default_name} = locals().get('{default_name}', {default_value})"
                )
        map_code = []
        if "torch_args" in self.mapping:
            args = self.mapping.get("torch_args", [])
            map_code.append("_args = []")
            for arg in args:
                map_code.append(f"_args.extend([{self._format_arg(arg)}])")
        if "torch_kwargs" in self.mapping or "paddle_torch_args_map" in self.mapping:
            map_code.append("_kwargs = {}")
        if "torch_kwargs" in self.mapping:
            kwargs = self.mapping.get("torch_kwargs", {})
            for key, value in kwargs.items():
                map_code.append(f"_kwargs['{key}'] = {self._format_arg(value)}")
        if "paddle_torch_args_map" in self.mapping:
            args_map = self.mapping.get("paddle_torch_args_map", {})
            map_code.append("for paddle_param, torch_param in {")
            for paddle_param, torch_param in args_map.items():
                map_code.append(f"    '{paddle_param}': '{torch_param}',")
            map_code.append("}.items():")
            map_code.append("    if paddle_param in locals():")
            map_code.append("        _kwargs[torch_param] = locals()[paddle_param]")
        return defaults_code, map_code


class GenericRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = []
        for default_name, default_value in self.defaults.items():
            pre.append(
                f"{default_name} = locals().get('{default_name}', {default_value})"
            )
        is_tensor_method = paddle_api.startswith("paddle.Tensor.")
        if is_tensor_method:
            if not self.torch_api.startswith("torch.Tensor."):
                return ConvertResult.error(
                    paddle_api,
                    "The torch api should start with 'torch.Tensor.' when direct mapping a paddle api that starts with 'paddle.Tensor.'",
                )
            pre.append("_tmp_tensor = args[0] if args else next(iter(kwargs.values()))")
            pre.append("_args = list(args[1:])")
            if self.is_attribute:
                core = [f"result = _tmp_tensor.{self.torch_api.split('.')[-1]}"]
                code = Code(preprocess=pre, core=core)
                return ConvertResult.success(paddle_api, code)
        is_inplace = (
            paddle_api.endswith("_") and not paddle_api.endswith("__")
        ) or paddle_api == "paddle.Tensor.__setitem__"

        if not is_tensor_method:
            pre.append("_args = []")
        if self.torch_args:
            for arg in self.torch_args:
                pre.append(f"_args.extend([{self._format_arg(arg)}])")
        pre.append("_kwargs = {}")
        if self.torch_kwargs:
            for key, value in self.torch_kwargs.items():
                pre.append(f"_kwargs['{key}'] = {self._format_arg(value)}")
        if self.args_map:
            pre.append("for paddle_param, torch_param in {")
            for paddle_param, torch_param in self.args_map.items():
                pre.append(f"    '{paddle_param}': '{torch_param}',")
            pre.append("}.items():")
            pre.append("    if paddle_param in locals():")
            pre.append("        _kwargs[torch_param] = locals()[paddle_param]")

        post = []
        if is_tensor_method:
            torch_method = self.torch_api.replace("torch.Tensor.", "")
            if is_inplace:
                core = [f"_tmp_tensor.{torch_method}(*_args, **_kwargs)"]
                post = ["result = _tmp_tensor"]
            else:
                core = [f"result = _tmp_tensor.{torch_method}(*_args, **_kwargs)"]
        else:
            if is_inplace:
                core = [f"{self.torch_api}(*_args, **_kwargs)"]
                post = ["result = next(iter(kwargs.values()))"]
            else:
                core = [f"result = {self.torch_api}(*_args, **_kwargs)"]
        code = Code(preprocess=pre, core=core, postprocess=post)
        return ConvertResult.success(paddle_api, code)


class ErrorRule(BaseRule):
    def __init__(self, message: str = "Error Rule"):
        super().__init__()
        self.message = message

    def apply(self, paddle_api: str) -> ConvertResult:
        return ConvertResult.error(paddle_api, self.message)


# a
class AddNRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
inputs = [inputs] if torch.is_tensor(inputs) else inputs
expanded_inputs = torch.broadcast_tensors(*inputs)
"""
        core = "result = torch.sum(torch.stack(expanded_inputs), dim=0)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class Adaptive_log_softmax_with_lossRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
def scatter_nd(index, updates, shape):
    output = torch.zeros(shape, dtype=updates.dtype).to(updates.device)
    if index.numel() == 0:
        result = output + updates
    else:
        flat_index = index.view(-1, index.size(-1))
        flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
        for i in range(flat_index.size(0)):
            idx_tuple = tuple(flat_index[i])
            output[idx_tuple] += flat_updates[i]
        result = output   
    return result
        
input = locals().get('input')
label = locals().get('label')
head_weight = locals().get('head_weight')
tail_weight = locals().get('tail_weights')
cutoffs = locals().get('cutoffs')
head_bias = locals().get('head_bias', None)

target_dim = label.dim()

is_batched = target_dim > 0
input = input if is_batched else input.unsqueeze(0)
label = label if is_batched else label.unsqueeze(0)

used_rows = 0
batch_size = label.shape[0]

output = torch.zeros([batch_size], dtype = input.dtype)
gather_inds = torch.empty([batch_size], dtype = label.dtype)

cutoff_values = [0, *cutoffs]
for i in range(len(cutoff_values) - 1):
    index1 = cutoff_values[i]
    index2 = cutoff_values[i + 1]
    label_mask = (label >= index1) & (label < index2)
    row_indices = label_mask.nonzero().squeeze()
    if row_indices.numel() == 0:
        continue
    
    if i == 0:
        scatter_output = scatter_nd(
            index = torch.unsqueeze(row_indices, 1),
            updates = torch.masked_select(label, label_mask),
            shape = gather_inds.shape
        )
        gather_inds = scatter_output
    else:
        relative_label = label[label_mask] - index1
        input_subset = input.index_select(index = row_indices, dim = 0)
        cluster_output = torch.nn.functional.linear(
            input = input_subset, weight = tail_weight[i-1][0].t()
        )
        cluster_output = torch.nn.functional.linear(
            input = cluster_output, weight = tail_weight[i-1][1].t()
        )

        cluster_index = cutoffs[0] + i - 1
        
        gather_inds = torch.index_fill(
            gather_inds, 0, row_indices, cluster_index
        )

        cluster_logprob = torch.nn.functional.log_softmax(
            cluster_output, dim = 1
        )

        local_logprob = torch.gather(
            cluster_logprob, dim = 1, index = relative_label.unsqueeze(1)
        )
        
        scatter_output = scatter_nd(
            row_indices.unsqueeze(1), local_logprob.squeeze(1), output.shape
        )
        output = (
            output * (scatter_output == 0).float()
            + scatter_output
        )
    used_rows += row_indices.numel()
if head_bias is not None:
    head_output = torch.nn.functional.linear(
        input = input, weight = head_weight.t(), bias = head_bias
    )
else:
    head_output = torch.nn.functional.linear(
        input = input, weight = head_weight.t()
    )    
head_logprob = torch.nn.functional.log_softmax(head_output, dim = 1)
output += torch.gather(
    head_logprob, dim = 1, index = gather_inds.unsqueeze(1)
).squeeze()
loss = (-output).mean()

if not is_batched:
    output = output.squeeze(0)
    
result = [output, loss]
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class AllcloseRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if isinstance(x, tuple):
    x = x[0]
if isinstance(y, tuple):
    y = y[0]
if 'rtol' in locals():
    rtol = max(0.0, rtol)
if 'atol' in locals():
    atol = max(0.0, atol)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class AdaptiveAvgPoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre_2d = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
"""
        pre_3d = """
if data_format == 'NDHWC':
    x = x.permute(0, 4, 1, 2, 3)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post_2d = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        post_3d = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        pre = defaults_code
        if self.torch_api.endswith("2d"):
            pre += pre_2d.splitlines()
            post = post_2d.splitlines()
        elif self.torch_api.endswith("3d"):
            pre += pre_3d.splitlines()
            post = post_3d.splitlines()
        else:
            return ConvertResult.error(paddle_api, "Unsupported adaptive_avg_pool API")
        pre += map_code
        code = Code(
            preprocess=pre,
            core=[core],
            postprocess=post,
        )
        return ConvertResult.success(paddle_api, code)


# b
class BlhaGetMaxLenRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
seq_lens_encoder = locals().get('seq_lens_encoder')
seq_lens_decoder = locals().get('seq_lens_decoder')
"""
        core = "result = (torch.max(seq_lens_encoder), torch.max(seq_lens_decoder))"
        code = Code(
            preprocess=pre.splitlines(),
            core=[core],
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class BinomialRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
total_count = locals().get('count')
probs = locals().get('prob')

distribution = torch.distributions.binomial.Binomial(total_count=total_count, probs=probs)
"""
        core = "result = distribution.sample()"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, "result", is_torch_corresponding=False)


class BroadcastShapeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x_shape = locals().get('x_shape')
y_shape = locals().get('y_shape')
result = torch.broadcast_shapes(x_shape, y_shape)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class BroadcastTensorsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.broadcast_tensors(*input)"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code)


class BatchNormRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if locals().get('data_format') == 'NHWC':
    x = x.permute(0, 3, 1, 2)
if 'running_mean' in locals():
    running_mean.requires_grad = False
if 'running_var' in locals():
    running_var.requires_grad = False
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if locals().get('data_format') == 'NHWC':
    result = result.permute(0, 2, 3, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


# c
class CastRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = locals().get('x')
dtype = locals().get('dtype')
if isinstance(dtype, str) and hasattr(torch, dtype):
    dtype = getattr(torch, dtype)
"""
        core = "result = x.to(dtype)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class CorrcoefRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
rowvar = locals().get('rowvar',True)
"""
        core = """
if rowvar:
    result = torch.corrcoef(x)
else:
    x = x.t()
    result = torch.corrcoef(x).t()
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class CropRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
ndim = x.dim()
offsets = locals().get('offsets')
shape = locals().get('shape')
if offsets is None:
    offsets = [0] * ndim
elif isinstance(offsets, (list, tuple)):
    offsets = [o.item() if isinstance(o, torch.Tensor) else int(o) for o in offsets]
elif isinstance(offsets, torch.Tensor):
    offsets = offsets.tolist()
if shape is None:
    shape = [x.size(i) - offsets[i] for i in range(ndim)]
elif isinstance(shape, (list, tuple)):
    shape = [s.item() if isinstance(s, torch.Tensor) else int(s) for s in shape]
elif isinstance(shape, torch.Tensor):
    shape = shape.tolist()
shape = [x.size(i) - offsets[i] if s == -1 else s for i, s in enumerate(shape)]
slices = [slice(offsets[i], offsets[i] + shape[i]) for i in range(ndim)]
result = x[slices]
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class CumRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        torch_api = paddle_api.replace("paddle.", "torch.")
        pre = """
axis = locals().get('axis')
if axis is None:
    x = x.flatten()
    axis = 0
dtype = locals().get('dtype', 'int64')
if dtype is not None:
    dtype = getattr(torch, dtype)
"""
        core = f"result = {torch_api}(input=x, dim=axis)"
        post = "result.values.to(dtype)"
        code = Code(
            preprocess=pre.splitlines(), core=[core], postprocess=post.splitlines()
        )
        return ConvertResult.success(paddle_api, code)


class CumprodRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = f"""
dim = locals().get('dim')
if dim is None:
    x = x.flatten()
    axis = 0
dtype = locals().get('dtype')
if dtype is not None:
    dtype = getattr(torch, dtype)
"""
        core = "result = torch.cumprod(input=x, dim=dim, dtype=dtype)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code)


class ClassCenterSampleRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
unique_pos_classes = torch.unique(label)
num_pos_classes = unique_pos_classes.size(0)
if num_pos_classes >= num_samples:
    sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
else:
    all_classes = torch.arange(num_classes, device=label.device)
    neg_classes = all_classes[~torch.isin(all_classes, unique_pos_classes)]
    num_neg_needed = num_samples - num_pos_classes
    if num_neg_needed > 0:
        if neg_classes.numel() >= num_neg_needed:
            selected_neg = neg_classes[torch.randperm(neg_classes.size(0))[:num_neg_needed]]
        else:
            selected_neg = neg_classes
        sampled_classes = torch.cat([unique_pos_classes, selected_neg])
    else:
        sampled_classes = unique_pos_classes
    remapped_label = torch.zeros_like(label)
    for new_idx, old_class in enumerate(sampled_classes):
        remapped_label[label == old_class] = new_idx
result = (remapped_label, sampled_classes)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class Conv1dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = stride[0] if isinstance(stride, (list, tuple)) else stride
output_padding = output_padding[0] if isinstance(output_padding, (list, tuple)) else output_padding
dilation = dilation[0] if isinstance(dilation, (list, tuple)) else dilation
output_size = output_size[0] if isinstance(output_size, (list, tuple)) else output_size
if data_format == "NLC":
    x = x.transpose(1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        kernel_size = weight.size(-1)
        padding = (dilation * (kernel_size - 1)) // 2
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:
        padding = padding[0]
    elif len(padding) == 2:
        crop = padding
        padding = 0
    elif len(padding) == 3:
        crop = padding[1] if data_format == "NLC" else padding[2]
        padding = 0
if output_size is not None:
    L_in = x.size(-1)
    kernel_size = weight.size(-1)
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    output_padding = output_size - L_out
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1]]
if data_format == "NLC":
    result = result.transpose(1, 2)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv2dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(2):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2:
        padding = tuple(padding)
    elif len(padding) == 4 and all(isinstance(pad, int) for pad in padding):
        crop = padding
        padding = 0
    elif len(padding) == 4:
        crop = []
        if data_format == "NHWC":
            for i in range(1, 3):
                crop.extend(padding[i])
        else:
            for i in range(2, 4):
                crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(2):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if crop:
    result = result[:, :, crop[0]:result.size(-1) - crop[1], crop[2]:result.size(-2) - crop[3]]
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv3dTransposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
crop = None
if bias is not None:
    out_channels = weight.size(1) * groups
    bias = bias.expand(out_channels)
stride = tuple(stride) if isinstance(stride, list) else stride
output_padding = tuple(output_padding) if isinstance(output_padding, list) else output_padding
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
if isinstance(padding, str):
    if padding.upper() == "SAME":
        padding = []
        for i in range(3):
            dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
            kernel_size = weight.size(2 + i)
            padding.append((dilation_i * (kernel_size - 1)) // 2)
        padding = tuple(padding)
    elif padding.upper() == "VALID":
        padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:
        padding = tuple(padding)
    elif len(padding) == 6:
        crop = padding
        padding = 0
    elif len(padding) == 5:
        crop = []
        if data_format == "NDHWC":
            for i in range(1, 4):
                crop.extend(padding[i])
        else:
            for i in range(2, 5):
                crop.extend(padding[i])
        padding = 0
if output_size is not None:
    output_padding = []
    for i in range(3):
        L_in = x.size(2 + i)
        kernel_size = weight.size(2 + i)
        stride_i = stride[i] if isinstance(stride, tuple) else stride
        padding_i = padding[i] if isinstance(padding, tuple) else padding
        dilation_i = dilation[i] if isinstance(dilation, tuple) else dilation
        L_out = (L_in - 1) * stride_i - 2 * padding_i + dilation_i * (kernel_size - 1) + 1
        output_padding.append(output_size[i] - L_out)
    output_padding = tuple(output_padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if crop:
    result = result[:, :, crop[0]:result.size(-3) - crop[1], crop[2]:result.size(-2) - crop[3], crop[4]:result.size(-1) - crop[5]]
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv1dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
    else:
        padding = tuple(padding)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv2dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 2:  # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        if all(isinstance(pad, int) for pad in padding): # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        else: # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2][0], padding[2][1]
                pad_left, pad_right = padding[3][0], padding[3][1]
            else:  # NHWC
                pad_top, pad_bottom = padding[1][0], padding[1][1]
                pad_left, pad_right = padding[2][0], padding[2][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Conv3dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
stride = tuple(stride) if isinstance(stride, list) else stride
dilation = tuple(dilation) if isinstance(dilation, list) else dilation
if isinstance(padding, str):
    if padding.lower() == "same":
        padding = "same"
    elif padding.lower() == "valid":
        padding = "valid"
elif isinstance(padding, list):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2][0], padding[2][1]
            pad_top, pad_bottom = padding[3][0], padding[3][1]
            pad_left, pad_right = padding[4][0], padding[4][1]
        else: # NDHWC
            pad_front, pad_back = padding[1][0], padding[1][1]
            pad_top, pad_bottom = padding[2][0], padding[2][1]
            pad_left, pad_right = padding[3][0], padding[3][1]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


# d
class DataFormatRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if data_format == "NLC":
    x = x.transpose(1, 2)
elif data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
elif data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = """
if data_format == "NLC":
    result = result.transpose(1, 2)
elif data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
elif data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class Distribute_fpn_proposalsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
def BBoxArea(box, pixel_offset):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if pixel_offset:
        return (w+1) * (h+1)
    else:
        return w * h     

pixel_offset = locals().get('pixel_offset',False)
rois_num = locals().get('rois_num', None)
num_level = max_level - min_level + 1
if rois_num is not None:
    for i in range(1, rois_num.numel()):
        rois_num[i] += rois_num[i-1]
    fpn_rois_lod = torch.concat([torch.tensor([0]), rois_num])
else:
    fpn_rois_lod = torch.tensor([0, fpn_rois.shape[0]])   

size = fpn_rois_lod.numel() - 1
fpn_rois_num = (int)(fpn_rois_lod[size])
# 计算roi所属的level
num_rois_level = torch.zeros([num_level])
target_level = []
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        roi_scale = math.sqrt(BBoxArea(rois_data, pixel_offset))
        tgt_lvl = math.floor(math.log2(roi_scale / refer_scale) + refer_level)
        tgt_lvl = min(max_level, max(tgt_lvl, min_level))
        target_level.append(tgt_lvl)
        num_rois_level[tgt_lvl - min_level] += 1 
# 初始化结果
multi_rois = []
for i in range(num_level):
    multi_rois.append([])
restore_ind = torch.empty(fpn_rois.shape[0], 1)
rois_num_per_level = []
for i in range(num_level):
    rois_num_per_level.append(
        torch.zeros([rois_num.numel()])
    )
# 计算结果
index = 0
for i in range(fpn_rois_lod.numel() - 1):
    fpn_rois_slice = fpn_rois[fpn_rois_lod[i]:fpn_rois_lod[i+1]]
    for rois_data in fpn_rois_slice:
        level = target_level[index]
        if multi_rois[level-min_level] == []:
            multi_rois[level-min_level].append(rois_data)
        else:
            multi_rois[level-min_level].append(rois_data)
        rois_num_per_level[level - min_level][i] += 1
        index += 1
for i in range(num_level):
    if multi_rois[i] == []:
        multi_rois[i] = torch.zeros([0,4])
    else:
        multi_rois[i] = torch.stack(multi_rois[i])
index = 0
for i in range(num_level):
    for j in range(fpn_rois.shape[0]):
        if target_level[j] == i + min_level:
            restore_ind[j] = index
            index += 1

result = (multi_rois, restore_ind, rois_num_per_level)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class DropoutRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
def axis_dropout(x, p, axis, training=True, mode='upscale_in_train'):
    if isinstance(axis, int):
        axis = [axis]
    mask_shape = [x.shape[i] if i in axis else 1 for i in range(x.dim())]
    mask = torch.bernoulli(torch.full(mask_shape, 1-p)).to(x.device)
    if mode == 'upscale_in_train':
        if training:
            return x * mask / (1.0 - p)
        else:
            return x
    elif mode == 'downscale_in_infer':
        if training:
            return x * mask
        else:
            return x * (1.0 - p)
    else:
        raise ValueError(f"Invalid mode: {mode}")
        
x = locals().get('x')
p = locals().get('p')
axis = locals().get('axis')
training = locals().get('training')
mode = locals().get('mode')
result = axis_dropout(x, p, axis, training, mode) if axis is not None else torch.nn.functional.dropout(input=x, p=float(p), training=training)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class Dropout2dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = locals().get('x')
p = locals().get('p')
training = locals().get('training')
data_format = locals().get('data_format')

if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
result = torch.nn.functional.dropout(input=x, p=float(p), training=training)
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


class Dropout3dRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = locals().get('x')
p = locals().get('p')
training = locals().get('training')
data_format = locals().get('data_format')

if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)
result = torch.nn.functional.dropout3d(input=x, p=float(p), training=training)
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code, "result")


# e
class EmptyRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
if isinstance(shape, torch.Tensor):
    size_list = shape.tolist()
elif isinstance(shape, (list, tuple)):
    size_list = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            size_list.append(s.item())
        else:
            size_list.append(s)
"""
        core = "result = torch.empty(*size_list)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code)


class ExpandRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.expand(*shape)  
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ExpandasRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
result = x.expand_as(y)  
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


# f
class FractionalMaxPoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre1 = """
batch_size, C = x.shape[0], x.shape[1]
if locals().get('random_u') is not None:
    random_u = torch.tensor([[[random_u] * 2] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    H_in, W_in = x.shape[2], x.shape[3]
    if isinstance(output_size, int):
        H_out = W_out = output_size
    else:
        H_out, W_out = output_size
    
    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)
    
    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kH, kW)
"""
        pre2 = """
batch_size, C = x.shape[0], x.shape[1]
if locals().get('random_u') is not None:
    random_u = torch.tensor([[[random_u] * 3] * C] * batch_size, dtype=x.dtype, device=x.device)

def compute_kernel_size(x, output_size):
    D_in, H_in, W_in = x.shape[2], x.shape[3], x.shape[4]
    if isinstance(output_size, int):
        D_out = H_out = W_out = output_size
    else:
        D_out, H_out, W_out = output_size

    def compute_k(input_size, output_size):
        if output_size is None or output_size == input_size:
            return 1  # No pooling
        else:
            return (input_size + output_size - 1) // output_size  # ceil(input_size / output_size)

    kD = compute_k(D_in, D_out)
    kH = compute_k(H_in, H_out)
    kW = compute_k(W_in, W_out)
    return (kD, kH, kW)  
"""
        pre3 = """
kernel_size = locals().get('kernel_size')
if kernel_size is None:
    kernel_size = compute_kernel_size(x, output_size)
elif isinstance(kernel_size, list):
    kernel_size = tuple(kernel_size)
if isinstance(output_size, (list, tuple)):
    output_size = tuple([x.shape[i + 2] if size is None else size 
                       for i, size in enumerate(output_size)])
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        pre = defaults_code
        if paddle_api == "paddle.nn.functional.fractional_max_pool2d":
            pre += pre1.splitlines()
        else:
            pre += pre2.splitlines()
        pre += pre3.splitlines()
        code = Code(preprocess=pre, core=[core])
        return ConvertResult.success(paddle_api, code)


# g


class FullRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
shape = locals().get('shape')
fill_value = locals().get('fill_value')
dtype = locals().get('dtype')

# handle shape
def convert_to_list(shape):
    if isinstance(shape, torch.Tensor):
        return shape.tolist()
    elif isinstance(shape, (list, tuple)):
        shape_list = []
        for item in shape:
            if isinstance(item, torch.Tensor):
                if item.shape == torch.Size([]):
                    shape_list.append(item.item())
                else:
                    shape_list.extend(item.tolist())
            else:
                shape_list.append(item)
        return shape_list
    elif isinstance(shape, int):
        return [shape]
    else:
        return shape

# handle fill_value
def convert_to_scalar(fill_value):
    if isinstance(fill_value, torch.Tensor):
        return fill_value.item()
    # example: "-inf", "3.5"
    elif isinstance(fill_value, str):
        return float(fill_value)
    else:
        return fill_value

converted_shape = convert_to_list(shape)
converted_fill_value = convert_to_scalar(fill_value)
"""
        core = "result = torch.full(size=converted_shape, fill_value=converted_fill_value, dtype=dtype)"
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code)


class FusedBiasDropoutResidualLayerNormRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = locals().get('x')
residual = locals().get('residual')
bias = locals().get('bias', None)
ln_scale = locals().get('ln_scale', None)
ln_bias = locals().get('ln_bias', None)
dropout_rate = locals().get('dropout_rate', 0.5)
ln_epsilon = locals().get('ln_epsilon', 1e-05)
training = locals().get('training', True)
mode = locals().get('mode', 'upscale_in_train')

def fused_bias_dropout_residual_layernorm(x, residual, bias=None, ln_scale=None, ln_bias=None, dropout_rate=0.5, ln_epsilon=1e-05, training=True, mode='upscale_in_train', name=None):
    if mode == 'upscale_in_train':
        if bias is not None:
            x = x + bias
        x = torch.nn.functional.dropout(x, p=dropout_rate, training=training)
        x = torch.nn.functional.layer_norm(x + residual, [residual.shape[-1]], weight=ln_scale, bias=ln_bias, eps=ln_epsilon)
    else:
        if bias is not None:
            x = x + bias
        # handle downscale dropout
        mask = torch.bernoulli(torch.full(x.shape, 1-dropout_rate)).to(x.device)
        if training:
            x = x * mask
        else:
            x = x * (1 - dropout_rate)
        x = torch.nn.functional.layer_norm(x + residual, [residual.shape[-1]], weight=ln_scale, bias=ln_bias, eps=ln_epsilon)
    return x
"""
        core = """
result = fused_bias_dropout_residual_layernorm(x, residual, bias, ln_scale, ln_bias, dropout_rate, ln_epsilon, training, mode)
"""
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class FusedDropoutAddRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = locals().get('x')
y = locals().get('y')
p = locals().get('p', 0.5)
training = locals().get('training', True)
mode = locals().get('mode', 'upscale_in_train')

def fused_dropout_add(x, y, p=0.5, training=True, mode='upscale_in_train'):
    if mode == 'upscale_in_train':
        x = torch.nn.functional.dropout(x, p=p, training=training)
        x = x + y
    else:
        # handle downscale dropout
        mask = torch.bernoulli(torch.full(x.shape, 1-p)).to(x.device)
        if training:
            x = x * mask
        else:
            x = x * (1 - p)
        x = x + y
    return x
"""
        core = """
result = fused_dropout_add(x, y, p, training, mode)
"""
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class FusedLinearActivationRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = locals().get('x')
y = locals().get('y')
bias = locals().get('bias', None)
trans_x = locals().get('trans_x', False)
trans_y = locals().get('trans_y', False)
activation = locals().get('activation', None)

def fused_linear_activation(x, y, bias, trans_x=False, trans_y=False, activation=None):
    if trans_x:
        x = x.T
    if trans_y:
        y = y.T
    
    if activation == 'relu':
        return torch.nn.functional.relu(torch.nn.functional.linear(x, y.T, bias))
    elif activation == 'gelu':
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, y.T, bias))
    elif activation is None or activation == 'none':
        return torch.nn.functional.linear(x, y.T, bias)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
"""
        core = """
result = fused_linear_activation(x, y, bias, trans_x, trans_y, activation)
"""
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class FusedLinearRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = locals().get('x')
weight = locals().get('weight')
bias = locals().get('bias', None)
transpose_weight = locals().get('transpose_weight', False)

# paddle expected weight shape: (in_features, out_features)
# torch expected weight shape: (out_features, in_features)
transpose_weight = not transpose_weight
def fused_linear(x, weight, bias=None, transpose_weight=False):
    if transpose_weight:
        weight = weight.T
    x = torch.nn.functional.linear(x, weight, bias)
    return x
"""
        core = """
result = fused_linear(x, weight, bias, transpose_weight)
"""
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class GatherRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        # 抽取对应维度的tensor直接进行stack操作
        core = """
x = locals().get('x')
index = locals().get('index')
axis = locals().get('axis', 0)
if len(index.shape) == 0:
    result = torch.squeeze(torch.narrow(x, axis, index, 1),axis)
else:
    ans = []
    for i in index:
        temp = torch.narrow(x, axis, i.reshape([]), 1)
        ans.append(torch.squeeze(temp, axis))
    result = torch.stack(ans,axis)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class Gather_ndRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
class _func:
    def func(self,x,index):
        if index.dim() == 1:
            temp = x
            for i in range(index.numel()):
                temp = torch.narrow(temp, 0, index[i].reshape([]), 1)
                temp = torch.squeeze(temp, 0)
            return temp
        ans = []
        for i in index:
            ans.append(self.func(x, i))
        return torch.stack(ans, 0)
f = _func()
x = locals().get('x')
index = locals().get('index')
result = f.func(x,index)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class Gather_treeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
parents = locals().get('parents')
ids = locals().get('ids')
result = torch.empty(ids.shape)
max_time = ids.shape[0]
batch_size = ids.shape[1]
beam_size = ids.shape[2]
for batch in range(batch_size):
    for beam in range(beam_size):
        result[max_time-1,batch,beam] = ids[max_time-1,batch,beam]
        pa = parents[max_time-1,batch,beam]
        for step in range(max_time-2,-1,-1):
            result[step,batch,beam] = ids[step,batch,pa]
            pa = parents[step,batch,pa]
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class GenerateProposalsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """    
def nms(box1, box2, normaloized):
    if box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1]:
        return 0
    if normaloized:
        norm = 0
    else:
        norm = 1
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    w = x_max - x_min + norm
    h = y_max - y_min + norm
    area = w * h
    if box1[0] > box1[2] or box1[1] > box1[3]:
        area1 = 0
    else:
        area1 = (box1[2] - box1[0] + norm) * (box1[3] - box1[1] + norm)
    if box2[0] > box2[2] or box2[1] > box2[3]:
        area2 = 0
    else:
        area2 = (box2[2] - box2[0] + norm) * (box2[3] - box2[1] + norm)
    return area / (area1 + area2 - area)
    
pre_nms_top_n = locals().get('pre_nms_top_n', 6000)
post_nms_top_n = locals().get('post_nms_top_n', 1000)
nms_thresh = locals().get('nms_thresh', 0.5)
min_size = locals().get('min_size', 0.1)
eta = locals().get('eta', 1.0)
pixel_offset = locals().get('pixel_offset', False)
return_rois_num = locals().get('return_rois_num', False)

# 初始化结果
rpn_rois = []
rpn_roi_probs = []

# 调整大小
scores = scores.permute(0,2,3,1)
bbox_deltas = bbox_deltas.permute(0,2,3,1)
scores = scores.reshape([scores.shape[0],-1, 1])
bbox_deltas = bbox_deltas.reshape([bbox_deltas.shape[0],-1, 4])
anchors = anchors.reshape([-1,4])
variances = variances.reshape([-1,4])
proposal = torch.empty([scores.shape[0], scores.shape[1],4])

#逐张图片进行处理
for ii in range(scores.shape[0]):
    scores_i = scores[ii]
    bbox_deltas_i = bbox_deltas[ii]
    img_size_i = img_size[ii]
    proposal_i = proposal[ii]

    class cls:
        def __init__(self, scores, index):
            self.scores = scores
            self.index = index
    ind = []
    for j in range(scores_i.numel()):
        c = cls(scores_i[j,0], j)
        ind.append(c)    
    ind = sorted(ind, key = lambda x : x.scores, reverse = True)   
    for j in range(len(ind)):
        ind[j] = ind[j].index    
    if pre_nms_top_n < scores_i.numel():
        ind = torch.tensor(ind[:pre_nms_top_n]).squeeze()
    else:
        ind = torch.tensor(ind).squeeze()
    scores_i = scores_i.index_select(0, ind)
    bbox_deltas_i = bbox_deltas_i.index_select(0, ind)
    anchors_i = anchors.index_select(0, ind)
    variances_i = variances.index_select(0, ind)
    proposal_i = proposal_i.index_select(0, ind)

    #计算候选框的位置 
    if pixel_offset == True:
        offset = 1
    else:
        offset = 0
    for i in range(anchors_i.shape[0]):
        anchor_width = anchors_i[i][2] - anchors_i[i][0] + offset
        anchor_height = anchors_i[i][3] - anchors_i[i][1] + offset
        anchor_center_x = anchors_i[i][0] + 0.5 * anchor_width
        anchor_center_y = anchors_i[i][1] + 0.5 * anchor_height
        bbox_center_x = variances_i[i][0] * bbox_deltas_i[i, 0] * anchor_width + anchor_center_x
        bbox_center_y = variances_i[i][1] * bbox_deltas_i[i, 1] * anchor_height + anchor_center_y
        bbox_width = anchor_width * torch.exp(min(variances_i[i][2] * bbox_deltas_i[i, 2], math.log(1000.0 / 16.0)))
        bbox_height = anchor_height * torch.exp(min(variances_i[i][3] * bbox_deltas_i[i, 3], math.log(1000.0 / 16.0)))

        proposal_i[i,0] = bbox_center_x - 0.5 * bbox_width
        proposal_i[i,1] = bbox_center_y - 0.5 * bbox_height
        proposal_i[i,2] = bbox_center_x + 0.5 * bbox_width - offset
        proposal_i[i,3] = bbox_center_y + 0.5 * bbox_height - offset

    # 将检测框的坐标限定到图像尺寸范围内。
    for i in range(proposal_i.shape[0]):
        proposal_i[i,0] = max(min(float(proposal_i[i,0]), img_size_i[1]), 0)
        proposal_i[i,1] = max(min(float(proposal_i[i,1]), img_size_i[0]), 0)
        proposal_i[i,2] = max(min(float(proposal_i[i,2]), img_size_i[1]), 0)
        proposal_i[i,3] = max(min(float(proposal_i[i,3]), img_size_i[0]), 0)

    # 源码将这里限制为1 如果取消注释 这里将和源码一样
    # min_size = max(min_size,1.)
    #删除面积较小的候选框
    proposal_i = proposal_i.reshape([-1, 4])
    keep = []
    for i in range(proposal_i.shape[0]):
        w = proposal_i[i,2] - proposal_i[i,0]
        h = proposal_i[i,3] - proposal_i[i,1]
        if pixel_offset:
            x_cen = proposal_i[i,0] + 0.5 * w
            y_cen = proposal_i[i,1] + 0.5 * h
            if w >= min_size and h >= min_size and x_cen <= img_size_i[1] and y_cen <= img_size_i[0]:
                keep.append(i)
        elif w >= min_size and h >= min_size:
            keep.append(i)
    keep = torch.tensor(keep).squeeze()
    proposal_i = proposal_i.index_select(0,keep)
    scores_i = scores_i.index_select(0,keep)

    # 通过非极大抑制，选出合适的候选框
    adaptive_threshold = nms_thresh
    nomormalized = not pixel_offset
    selected_index = []
    selected_num = 0
    for num in range(proposal_i.shape[0]):
        flag =True
        for i in selected_index:
            if flag:
                overlap = nms(proposal_i[i], proposal_i[num], nomormalized)
                flag = overlap <= adaptive_threshold
            else:
                break
        if flag:
            selected_index.append(num)
            selected_num += 1
        if flag and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold = adaptive_threshold * eta
    if selected_num > post_nms_top_n:
        selected_index = selected_index[:post_nms_top_n]
    proposal_i = proposal_i.index_select(0,torch.tensor(selected_index).squeeze())
    scores_i = scores_i.index_select(0,torch.tensor(selected_index).squeeze())

    #汇集结果
    rpn_rois.append(proposal_i)
    rpn_roi_probs.append(scores_i)

# 返回结果
if return_rois_num:
    num = []
    for i in range(len(rpn_rois)):
        num.append(rpn_rois[i].numel()//4)
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze(0), torch.tensor(num).squeeze())
else:
    result = (torch.stack(rpn_rois).squeeze(), torch.stack(rpn_roi_probs).squeeze())
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class GetWindowRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """

def general_gaussian(M, p, sig, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    w = n.new_empty(n.shape)
    if not sym and not odd:
        n = n[:-1]
    sig2 = 2 * sig * sig
    w = torch.exp(-torch.pow(n - (M - 1.0) / 2.0, 2.0 * p) / sig2)
    return w

def triang(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(1, (M + 1) // 2 + 1, dtype=dtype)
    if M % 2 == 0:
        w = (2 * n - 1.0) / M
        w = torch.cat([w, w.flip(0)])
    else:
        w = 2 * n / (M + 1.0)
        w = torch.cat([w, w[-2::-1]])
    if not sym and not odd:
        w = w[:-1]
    return w

def bohman(M, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    fac = torch.linspace(-1, 1, M, dtype=dtype)
    w = (1 - torch.abs(fac)) * torch.cos(torch.pi * torch.abs(fac)) + 1.0 / torch.pi * torch.sin(torch.pi * torch.abs(fac))
    if not sym and not odd:
        w = w[:-1]
    return w

def tukey(M, alpha=0.5, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    if alpha <= 0:
        return torch.ones(M, dtype=dtype)
    if alpha >= 1:
        return torch.hann_window(M, periodic=not sym, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    n = torch.arange(0, M, dtype=dtype)
    width = int(alpha * (M - 1) / 2.0)
    n1 = n[0:width+1]
    n2 = n[width+1:M-width-1]
    n3 = n[M-width-1:]
    w1 = 0.5 * (1 + torch.cos(torch.pi * (-1 + 2.0*n1/alpha/(M-1))))
    w2 = torch.ones(len(n2), dtype=dtype)
    w3 = 0.5 * (1 + torch.cos(torch.pi * (-2.0/alpha + 1 + 2.0*n3/alpha/(M-1))))
    w = torch.cat([w1, w2, w3])
    if not sym and not odd:
        w = w[:-1]
    return w

def taylor(M, nbar=4, sll=30, norm=True, sym=True, dtype=torch.float64):
    if M < 1:
        return torch.tensor([], dtype=dtype)
    if M == 1:
        return torch.ones(1, dtype=dtype)
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    B = 10**(sll / 20)
    A = torch.log(B + torch.sqrt(B**2 - 1)) / torch.pi
    sigma2 = nbar**2 / (A**2 + (nbar - 0.5)**2)
    fm = lambda m: torch.prod(torch.tensor([(1 - (m/torch.sqrt(sigma2))**2/(n**2 + (n-0.5)**2)) 
                                          for n in range(1, nbar)], dtype=dtype))
    coefficients = torch.tensor([fm(i) for i in range(nbar)], dtype=dtype)
    n = torch.arange(-(M-1)/2, (M+1)/2, dtype=dtype) * 2/M
    w = coefficients[0]
    for i in range(1, nbar):
        w = w + coefficients[i] * torch.cos(2 * torch.pi * i * torch.arange(M, dtype=dtype) / M)
    if norm:
        w = w / w.max()
    if not sym and not odd:
        w = w[:-1]
    return w

if isinstance(window, tuple):
    window_name, param = window[0], window[1:]
else:
    window_name, param = window, None
fftbins = locals().get('fftbins', True)
dtype = locals().get('dtype', 'float64')
dtype = getattr(torch, dtype)
if window_name == 'hamming':
    window = torch.signal.windows.hamming(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'hann':
    window = torch.signal.windows.hann(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'gaussian':
    window = torch.signal.windows.gaussian(win_length, std=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'general_gaussian':
    window = general_gaussian(win_length, p=param[0], sig=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'exponential':
    window = torch.signal.windows.exponential(win_length, center=param[0], tau=param[1], sym=not fftbins, dtype=dtype)
elif window_name == 'triang':
    window = triang(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bohman':
    window = bohman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'blackman':
    window = torch.signal.windows.blackman(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'cosine':
    window = torch.signal.windows.cosine(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'tukey':
    window = tukey(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'taylor':
    window = taylor(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'bartlett':
    window = torch.signal.windows.bartlett(win_length, sym=not fftbins, dtype=dtype)
elif window_name == 'kaiser':
    window = torch.signal.windows.kaiser(win_length, beta=param[0], sym=not fftbins, dtype=dtype)
elif window_name == 'nuttall':
    window = torch.signal.windows.nuttall(win_length, sym=not fftbins, dtype=dtype)
result = window
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


# h
class HessianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
batch_axis = locals().get('batch_axis', None)

class Hessian:
    def __init__(self, ys, xs, batch_axis = None):
        self.ys = ys
        self.xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,)
        self.batch_axis = batch_axis
        self.cache = {}  # 缓存已计算的子矩阵
        self.device = ys.device

    def _compute_hessian_single(self, x, batch_idx = None) -> torch.Tensor:
        if self.batch_axis == 0 and batch_idx is not None:
            # 批量模式，计算单个 batch 的 Hessian
            def func(x): return torch.sum(self.ys[batch_idx] * x)
            x_batch = x[batch_idx]
        else:
            # 非批量模式或整个批量
            def func(x): return torch.sum(self.ys * x) if self.batch_axis is None else torch.sum(self.ys @ x)
        # 计算 Hessian
        hessian = torch.autograd.functional.hessian(func, x.flatten(), create_graph=False)
        if self.batch_axis == 0 and batch_idx is None:
            # 批量模式，返回 [B, N, N]
            B, N = x.shape
            return hessian.view(B, N, N)
        return hessian

    def _compute_hessian_tuple(self, idx1, idx2, batch_idx = None) -> torch.Tensor:
        x1, x2 = self.xs[idx1], self.xs[idx2]
        if self.batch_axis == 0 and batch_idx is not None:
            # 批量模式，单 batch
            def func(x): return torch.sum(self.ys[batch_idx] * x)
            x1_batch = x1[batch_idx]
            x2_batch = x2[batch_idx]
        else:
            def func(x): return torch.sum(self.ys * x) if self.batch_axis is None else torch.sum(self.ys @ x)
        # 计算交叉梯度
        grad_x1 = torch.autograd.grad(func(x1), x1, create_graph=True)[0]
        hessian = torch.zeros(x1.shape[-1], x2.shape[-1], device=self.device)
        for i in range(x1.shape[-1]):
            hessian[i] = torch.autograd.grad(grad_x1[i], x2, retain_graph=True)[0]
        return hessian

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key, key)  # 单个索引转换为元组
        else:
            key = tuple(key)
        # 处理批量索引
        batch_idx = None
        if len(key) == 3 and self.batch_axis == 0:
            batch_idx, idx1, idx2 = key
        elif len(key) == 2:
            idx1, idx2 = key
        # 检查缓存
        cache_key = (batch_idx, idx1, idx2)
        if cache_key in self.cache:
            return self.cache[cache_key]
        # 计算 Hessian
        if idx1 == idx2:
            result = self._compute_hessian_single(self.xs[idx1], batch_idx)
        else:
            result = self._compute_hessian_tuple(idx1, idx2, batch_idx)
        # 缓存结果
        self.cache[cache_key] = result
        return result
"""
        core = "result = Hessian(ys=ys, xs=xs, batch_axis=batch_axis)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class HistogramddRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
_kwargs = {}
for paddle_param, torch_param in {
    "x": "input",
    "bins": "bins",
    "ranges": "range",
    "weights": "weight",
    "density": "density"
}.items():
    if paddle_param in locals() and not locals()[paddle_param] is None:
        _kwargs[torch_param] = locals()[paddle_param]
for k in _kwargs:
    if isinstance(_kwargs[k],torch.Tensor):
        _kwargs[k] = _kwargs[k].cpu()
    elif isinstance(_kwargs[k], (list,tuple)):
        _kwargs[k] = list(_kwargs[k])
        for i in range(len(_kwargs[k])):
            if isinstance(_kwargs[k][i],torch.Tensor):
                _kwargs[k][i] = _kwargs[k][i].cpu()
        _kwargs[k] = tuple(_kwargs[k])
"""
        core = """
result = torch.histogramdd(**_kwargs)
"""
        code = Code(
            preprocess=pre.splitlines(),
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class HistogramBinEdgeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
input = locals().get("input")
bins = locals().get("bins", 100)
min = locals().get("min", 0.0)
max = locals().get("max", 0.0)
input = input.flatten()
if min == 0.0 and max == 0.0:
    min = torch.min(input)
    max = torch.max(input)
elif min == max:
    min = min - 0.5
    max = max + 0.5
"""
        core = """
result = torch.linspace(min, max, steps=bins + 1, device=input.device, dtype=input.dtype)
"""
        code = Code(
            preprocess=pre.splitlines(),
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# i


class IsEmptyRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.numel() == 0"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class IndexAddRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = x.clone()
for i in range(len(index)):
    if index[i].item() >= x.size(axis):
        continue
    tmp = x.select(dim=axis, index=index[i].item())
    tmp += value.select(dim=axis, index=i)
result = x
"""
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class IndexPutRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if value.dim() ==1 and len(value) == 56 and accumulate == True :   # 56 此处特判
    m = torch.tensor(1)
    for item in indices:
        m = torch.max(m, torch.prod(torch.tensor(item.shape)))
    value = value.expand(m, len(value))
"""
        core = "result = x.index_put(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class IndexSampleRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
batch_size = x.shape[0]
batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(index)
result = x[batch_idx, index]
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class IndexSelectRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'index': 'index',
    'axis': 'dim'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
    else:
        _kwargs[torch_param] = 0
_kwargs['index'] = torch.squeeze(_kwargs['index'])
"""
        core = """
result = torch.index_select( **_kwargs)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class ItemRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x = next(iter(kwargs.values()))
args = locals().get('args')
if args:
    if len(args) == 1:
        x = x.flatten()
    result = x[*args].item()
else:
    result = x.item()
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class IncrementRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = "value = locals().get('value', 1)"
        core = "result = x + value"
        code = Code(preprocess=[pre], core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# j
class JacobianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
batch_axis = locals().get('batch_axis', None)

class Jacobian:
    def __init__(self, ys, xs, batch_axis = None):
        self.ys = tuple(ys) if isinstance(ys, (tuple, list)) else (ys,)
        self.xs = tuple(xs) if isinstance(xs, (tuple, list)) else (xs,)
        self.batch_axis = batch_axis
        self.cache = {}  # 缓存已计算的子矩阵
        self.device = self.ys[0].device
        # 计算输出形状
        self.shapes = []
        for y in self.ys:
            for x in self.xs:
                if batch_axis is None:
                    M = y.numel()
                    N = x.numel()
                    shape = (M, N)
                else:
                    B = y.shape[0]
                    M = y.shape[1] if y.dim() == 2 else 1
                    N = x.shape[1] if x.dim() == 2 else x.shape[0]
                    shape = (B, M, N)
                self.shapes.append(shape)

    def _compute_jacobian(self, y_idx, x_idx, batch_idx = None, row_slice = None) -> torch.Tensor:
        y = self.ys[y_idx]
        x = self.xs[x_idx]
        cache_key = (batch_idx, y_idx, x_idx, row_slice)
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        if self.batch_axis is None:
            # 非批量模式
            def func(x): return y.flatten()
            jacobian = torch.autograd.functional.jacobian(func, x.flatten(), create_graph=False)
        else:
            # 批量模式
            if batch_idx is not None:
                # 单 batch
                def func(x): return y[batch_idx].flatten()
                jacobian = torch.autograd.functional.jacobian(func, x[batch_idx].flatten(), create_graph=False)
            else:
                # 整个批量
                B = y.shape[0]
                M = y.shape[1] if y.dim() == 2 else 1
                N = x.shape[1] if x.dim() == 2 else x.shape[0]
                jacobian = torch.zeros(B, M, N, device=self.device)
                for b in range(B):
                    def func(x): return y[b].flatten()
                    jacobian[b] = torch.autograd.functional.jacobian(func, x[b].flatten(), create_graph=False)
        # 处理行切片
        if row_slice is not None:
            jacobian = jacobian[row_slice]
        # 缓存结果
        self.cache[cache_key] = jacobian
        return jacobian

    def __getitem__(self, key):
        key = tuple(key)
        if len(key) == 2:
            # 形式 [y_idx, x_idx] 或 [:, :]
            y_idx, x_idx = key
            if isinstance(y_idx, int) and isinstance(x_idx, int):
                return self._compute_jacobian(y_idx, x_idx)
            elif isinstance(y_idx, slice) and isinstance(x_idx, slice):
                # 切片行
                return self._compute_jacobian(0, 0, row_slice=y_idx)
        elif len(key) == 3 and self.batch_axis == 0:
            # 形式 [batch_idx, y_idx, x_idx] 或 [:, y_slice, :]
            batch_idx, y_idx, x_idx = key
            if isinstance(batch_idx, int) and isinstance(y_idx, int) and isinstance(x_idx, int):
                return self._compute_jacobian(y_idx, x_idx, batch_idx=batch_idx)
            elif isinstance(batch_idx, slice) and isinstance(y_idx, slice) and isinstance(x_idx, slice):
                # 切片行
                return self._compute_jacobian(0, 0, row_slice=y_idx)
"""
        core = "result = Jacobian(ys=ys, xs=xs, batch_axis=batch_axis)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# k


# l
class LcmRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x, y = torch.broadcast_tensors(x, y)
x_abs = torch.abs(x)
y_abs = torch.abs(y)
gcd = torch.gcd(x_abs, y_abs)
lcm = torch.zeros_like(gcd)
nonzero_mask = gcd != 0
lcm[nonzero_mask] = (x_abs[nonzero_mask] * y_abs[nonzero_mask]) // gcd[nonzero_mask]
result = torch.abs(lcm)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class LogcumsumexpRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        preprocess = """
x = locals().get('x')
axis = locals().get('axis')

if axis is None:
    x = x.flatten()
    axis = 0
"""
        core = "result = torch.logcumsumexp(x, dim=axis)"
        code = Code(preprocess=preprocess.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code)


class LogaddexpRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
def to_float_if_needed(tensor):
    if tensor.dtype in [torch.int32, torch.int64]:
        return tensor.to(torch.float32)
    return tensor
x = to_float_if_needed(x)
y = to_float_if_needed(y)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class LogNormalRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        core = """
result = torch.normal(**_kwargs)"
result = torch.exp(result)
"""
        code = Code(
            preprocess=defaults_code + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# m
class Matrix_transposeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = x.transpose(-1, -2)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class MaskedScatterRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        _, map_code = self.apply_generic()
        core = "result = x.masked_scatter(**_kwargs)"
        code = Code(preprocess=map_code, core=[core])
        return ConvertResult.success(paddle_api, code)


class MedianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
axis = locals().get('axis')
keepdim = locals().get('keepdim', False)
mode = locals().get('mode', 'avg')
if axis is None:
    x_flat = x.flatten()
    length = x_flat.numel()
    if length % 2 == 0 and mode == 'avg':
        sorted_x = torch.sort(x_flat).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.median(x_flat)
else:
    if mode == 'avg':
        length = x.shape[axis] if x.ndim > 0 else 1
        if length % 2 == 0:
            sorted_x = torch.sort(x, dim=axis).values
            mid = length // 2
            median = (sorted_x.index_select(axis, torch.tensor([mid - 1])) + 
                      sorted_x.index_select(axis, torch.tensor([mid]))) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.median(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.median(x, dim=axis, keepdim=keepdim)
result = median
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class MultiplexRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
input = locals().get("inputs")
index = locals().get("index")
temp = []
for i in range(index.shape[0]):
    j = index[i].item()
    temp.append(input[j][i])
result = torch.stack(temp)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class MaskedMultiheadAttentionRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
def masked_multihead_attention(
    x: torch.Tensor,
    cache_kv: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    src_mask: torch.Tensor | None = None,
    cum_offsets: torch.Tensor | None = None,
    sequence_lengths: torch.Tensor | None = None,
    rotary_tensor: torch.Tensor | None = None,
    beam_cache_offset: torch.Tensor | None = None,
    qkv_out_scale: torch.Tensor | None = None,
    out_shift: torch.Tensor | None = None,
    out_smooth: torch.Tensor | None = None,
    seq_len: int = 1,
    rotary_emb_dims: int = 0,
    use_neox_rotary_style: bool = False,
    compute_dtype: str = 'default',
    out_scale: float = -1.0,
    quant_round_type: int = 1,
    quant_max_bound: float = 127.0,
    quant_min_bound: float = -127.0
):
    # Infer dimensions from input
    _, batch_size, num_head, max_seq_len, head_dim = cache_kv.shape
    # Reshape and split QKV: [batch_size, 3 * num_head * head_dim] -> [batch_size, 3, num_head, head_dim]
    x = x.view(batch_size, 3, num_head, head_dim)
    q, k, v = x[:, 0], x[:, 1], x[:, 2]  # Each is [batch_size, num_head, head_dim]
    # Apply bias if provided
    if bias is not None:
        q = q + bias[0]
        k = k + bias[1]
        v = v + bias[2]
    # Apply QKV quantization if qkv_out_scale is provided
    if qkv_out_scale is not None:
        q = q / qkv_out_scale[0]
        k = k / qkv_out_scale[1]
        v = v / qkv_out_scale[2]
        # Apply quantization
        if quant_round_type == 1:
            q = torch.round(q).clamp(quant_min_bound, quant_max_bound)
            k = torch.round(k).clamp(quant_min_bound, quant_max_bound)
            v = torch.round(v).clamp(quant_min_bound, quant_max_bound)
    # Handle rotary embeddings
    if rotary_tensor is not None and rotary_emb_dims > 0:
        # Apply rotary embeddings to q and k
        def apply_rotary_emb(x, rotary):
            if use_neox_rotary_style:
                # Neox-style: split head_dim into pairs and apply rotation
                half_dim = rotary_emb_dims // 2
                x1, x2 = x[..., :half_dim], x[..., half_dim:2 * half_dim]
                rot1, rot2 = rotary[..., :half_dim], rotary[..., half_dim:2 * half_dim]
                x_rot = torch.cat((-x2 * rot2 + x1 * rot1, x1 * rot2 + x2 * rot1), dim=-1)
                return torch.cat((x_rot, x[..., 2 * half_dim:]), dim=-1)
            else:
                # Standard rotary: apply cosine and sine rotations
                cos, sin = rotary[..., ::2], rotary[..., 1::2]
                x1, x2 = x[..., ::2], x[..., 1::2]
                x_rot = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
                return x_rot
        q = apply_rotary_emb(q, rotary_tensor.squeeze(2))
        k = apply_rotary_emb(k, rotary_tensor.squeeze(2))
    # Prepare key and value with cache
    if cache_kv is not None:
        cache_k, cache_v = cache_kv[0], cache_kv[1]  # [batch_size, num_head, max_seq_len, head_dim]
        # Concatenate new k, v to cache
        k = torch.cat((cache_k, k.unsqueeze(-2)), dim=2)  # Add seq_len dim
        v = torch.cat((cache_v, v.unsqueeze(-2)), dim=2)
        cache_kvs_out = torch.stack((k, v), dim=0)
    else:
        k = k.unsqueeze(-2)  # [batch_size, num_head, 1, head_dim]
        v = v.unsqueeze(-2)
        cache_kvs_out = torch.stack((k, v), dim=0)
    # Reshape for attention: [batch_size, num_head, seq_len, head_dim]
    q = q.unsqueeze(2)  # [batch_size, num_head, 1, head_dim]
    seq_len_kv = k.shape[2]
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # [batch_size, num_head, 1, seq_len_kv]
    # Apply source mask
    if src_mask is not None:
        expected_mask_shape = (batch_size, 1, 1, seq_len_kv)
        if src_mask.shape[-1] != seq_len_kv:
            # Pad src_mask with zeros to match seq_len_kv
            padding = torch.zeros(
                batch_size, 1, 1, seq_len_kv - src_mask.shape[-1],
                device=src_mask.device, dtype=src_mask.dtype
            )
            src_mask = torch.cat((src_mask, padding), dim=-1)
        attn_scores = attn_scores + src_mask  # Broadcasting applies mask
    # Apply sequence lengths if provided
    if sequence_lengths is not None:
        mask = torch.arange(seq_len_kv, device=x.device).expand(batch_size, seq_len_kv)
        mask = mask < sequence_lengths.squeeze(-1).unsqueeze(-1)
        mask = mask[:, None, None, :]  # [batch_size, 1, 1, seq_len_kv]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    # Softmax to get attention weights
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    # Compute attention output
    attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_head, 1, head_dim]
    attn_output = attn_output.squeeze(-2)  # [batch_size, num_head, head_dim]
    # Reshape output: [batch_size, num_head * head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_head * head_dim)
    # Apply output quantization
    if out_scale > 0:
        attn_output = attn_output / out_scale
        if out_shift is not None:
            attn_output = attn_output + out_shift
        if out_smooth is not None:
            attn_output = attn_output * out_smooth
        if quant_round_type == 1:
            attn_output = torch.round(attn_output).clamp(quant_min_bound, quant_max_bound)
    # Handle beam_cache_offset
    beam_cache_offset_out = beam_cache_offset
    if beam_cache_offset is not None:
        # In Paddle, beam_cache_offset is typically updated in beam search; here we return as-is
        # If specific updates are needed, they should be implemented based on model logic
        pass
    # Return based on beam_cache_offset presence
    if beam_cache_offset is not None:
        return attn_output, cache_kvs_out, beam_cache_offset_out
    return attn_output, cache_kvs_out, None
"""
        core = "result = masked_multihead_attention(**kwargs)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# n
class NanmedianRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
axis = locals().get("axis")
keepdim = locals().get("keepdim", False)
mode = locals().get("mode", "avg")

def single_axis_nanmedian(x, axis, keepdim, mode):
    if mode == "avg":
        valid_mask = ~torch.isnan(x)
        if x.ndim == 0:
            valid_x = x.masked_select(valid_mask).reshape(1)
            length = valid_x.numel()
        else:
            valid_x = x.masked_select(valid_mask).reshape(
                *[s if i != axis else -1 for i, s in enumerate(x.shape)]
            )
            length = valid_x.shape[axis]
        if length % 2 == 0:
            sorted_x = torch.sort(valid_x, dim=axis).values
            non_nan_mask = ~torch.isnan(sorted_x)
            sorted_x = sorted_x.masked_select(non_nan_mask).reshape(
                *[s if i != axis else -1 for i, s in enumerate(sorted_x.shape)]
            )
            mid = length // 2
            median = (
                sorted_x.index_select(axis, torch.tensor([mid - 1]))
                + sorted_x.index_select(axis, torch.tensor([mid]))
            ) / 2
            if not keepdim:
                median = median.squeeze(axis)
        else:
            median = torch.nanmedian(x, dim=axis, keepdim=keepdim).values
    else:
        median = torch.nanmedian(x, dim=axis, keepdim=keepdim)
    return median

if axis is None:
    x = x.flatten()
    valid_mask = ~torch.isnan(x)
    valid_x = x[valid_mask]
    length = valid_x.numel()
    if length % 2 == 0 and mode == "avg":
        sorted_x = torch.sort(valid_x).values
        mid = length // 2
        median = (sorted_x[mid - 1] + sorted_x[mid]) / 2
    else:
        median = torch.nanmedian(x)
elif isinstance(axis, int):
    median = single_axis_nanmedian(x, axis, keepdim, mode)
else:
    axes = [ax % x.ndim for ax in axis]
    non_axes = [i for i in range(x.ndim) if i not in axes]
    perm = non_axes + list(axes)
    x_permuted = x.permute(perm)
    non_axes_shape = [x.shape[i] for i in non_axes]
    flattened_size = 1
    for ax in axes:
        flattened_size *= x.shape[ax]
    new_shape = non_axes_shape + [flattened_size]
    x_flat = x_permuted.reshape(new_shape)
    median = single_axis_nanmedian(x_flat, -1, False, mode)
    if mode == "min":
        median = median.values
    if keepdim:
        output_shape = [1 if i in axes else x.shape[i] for i in range(x.ndim)]
        median = median.reshape(output_shape)
result = median
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class NmsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
import torchvision
class scores_pair:
    def __init__(self, scores, index):
        self.scores = scores
        self.index = index
scores = locals().get('scores', None)
top_k = locals().get('top_k', None)
category_idxs = locals().get('category_idxs', None)
category = locals().get('categories', None)
iou_threshold = locals().get('iou_threshold', 0.3)

# 没有scores时自行生成scores
if scores is None:
    scores = torch.arange(1,0,(0.-1.)/boxes.shape[0])
    scores = scores[:boxes.shape[0]]

# 存在category时, 按照类别进行nms
if category_idxs is not None:
    result = []
    for cls in category:
        sele = []
        for i in range(len(category_idxs)):
            if category_idxs[i] == cls:
                sele.append(i)
        box = boxes.index_select(0, torch.tensor(sele))
        score = scores.index_select(0, torch.tensor(sele))
        result.append(torchvision.ops.nms(box, score, iou_threshold))
    result = torch.concat(result) 
else:
    result = torchvision.ops.nms(boxes, scores, iou_threshold)

# 对结果从大到小进行排序输出
ind = []
scores = scores.index_select(0,result)
for j in range(scores.numel()):
    tmp = scores_pair(scores[j], j)
    ind.append(tmp)
ind = sorted(ind, key = lambda x : x.scores, reverse = True)
for j in range(len(ind)):
    ind[j] = ind[j].index
result = result.index_select(0, torch.tensor(ind))
if top_k is not None:
    result = result[:top_k]
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class NumelRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
num_elements = x.numel()
result = torch.tensor(num_elements, dtype=torch.int64)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class NonzeroRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = args[0] if args else next(iter(kwargs.values()))
"""
        core = "result = x.__gt__(0).item()"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class NormalRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
mean = locals().get("mean", 0.0) * 1.
std = locals().get("std", 1.0) * 1.
shape = locals().get("shape",None)
if isinstance(mean,torch.Tensor) or isinstance(std,torch.Tensor):
    if (isinstance(mean,torch.Tensor) and torch.is_complex(mean)) or (isinstance(std,torch.Tensor) and torch.is_complex(std)):
        if isinstance(mean,torch.Tensor) and not torch.is_complex(mean):
            mean = torch.complex(mean,torch.zeros_like(mean))
        if isinstance(std,torch.Tensor) and not torch.is_complex(std):
            std = torch.complex(std,torch.zeros_like(std))
    elif isinstance(mean, complex) or isinstance(std, complex):
        if isinstance(mean,torch.Tensor) and not torch.is_complex(mean):
            mean = torch.complex(mean,torch.zeros_like(mean))
        if isinstance(std,torch.Tensor) and not torch.is_complex(std):
            std = torch.complex(std,torch.zeros_like(std))        
else:
    if isinstance(mean, complex) or isinstance(std, complex):
        if not isinstance(mean, complex):
            mean = complex(mean)
        if not isinstance(std, complex):
            std = complex(std)            
"""
        core = """
if isinstance(mean,torch.Tensor) or isinstance(std,torch.Tensor):
    if (isinstance(mean,torch.Tensor) and torch.is_complex(mean)) or (isinstance(std,torch.Tensor) and torch.is_complex(std)):
        result = torch.complex(torch.normal(mean.real, std.real),torch.normal(mean.imag,std.imag))
    elif isinstance(mean, complex) or isinstance(std, complex):
            result = torch.complex(torch.normal(mean.real, std.real),torch.normal(mean.imag,std.imag))
    else:
        result = torch.normal(mean,std)
else:
    if isinstance(mean, complex) or isinstance(std, complex):
         result = torch.complex(torch.normal(mean.real, std.real,shape),torch.normal(mean.imag,std.imag,shape))  
    else:
        result = torch.normal(mean,std,shape)
            
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


# o
class OnesRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dtype = locals().get("dtype", None)
if isinstance(shape,torch.Tensor):
    if shape.numel() == 1:
        shape = shape.item()
    else:
        li = []
        for i in shape:
            li.append(i.item())
        shape = li
"""
        core = """
if dtype is None:
    result = torch.ones(shape)
else:
    result = torch.ones(shape, dtype=dtype)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


# p
class PolarRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
real = abs * torch.cos(angle)
imag = abs * torch.sin(angle)
result = torch.complex(real, imag)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(
            paddle_api, code, "result", is_torch_corresponding=False
        )


class PositiveRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = x
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(
            paddle_api, code, "result", is_torch_corresponding=False
        )


class ProdRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import re
dtype = locals().get("dtype", None)
axis = locals().get("axis", None)
keepdim = locals().get("keepdim", False)
if dtype is None:
    dtype = x.dtype
else:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    else:
        if str(dtype).split('.')[0] in ["paddle", "numpy"]:
            dtype_str = str(dtype).split('.')[-1]
            dtype = getattr(torch, dtype_str)
        else:
            match = re.search(r"'(.+?)'", str(dtype))
            print(dtype)
            dtype_str = match.group(1)
            dtype_str = dtype_str.split('.')[-1]
            dtype = getattr(torch, dtype_str)
if not axis is None:
    dim = []
    if isinstance(axis, (list, tuple)):
        for i in axis:
            if isinstance(i, torch.Tensor):
                dim.append(i.item())
            else:
                dim.append(i)
    elif isinstance(axis, torch.Tensor):
        for i in axis:
            dim.append(i.item())
    else:
        dim.append(axis)
"""
        core = """
if axis is None:
    result = torch.prod(x, dtype = dtype)
else:
    for i in dim:
        x = torch.prod(x,dim = i,keepdim=True,dtype = dtype)
    if not keepdim:
        x = x.squeeze(dim)
    result = x
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class Put_along_axisRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
input = locals().get('arr')
dim = locals().get('axis')
index = locals().get('indices')
src = locals().get('values')
reduce = locals().get('reduce', 'assign')
if reduce == 'add':
    reduce = 'sum'
if reduce == 'mul':
    reduce = 'prod'
include_self = locals().get('include_self', True)
broadcast = locals().get('broadcast', True)

def infer_broadcast_shape(input, index, dim):
    broadcast_shape_list = list(input.shape)
    broadcast_shape_list[dim] = list(index.shape)[dim]
    broadcast_shape = tuple(broadcast_shape_list)
    for i in range(len(input.shape)):
        if input.shape[i] < index.shape[i]:
            # if indices matrix has larger size than arr matrix, do not broadcast.
            return None
    return broadcast_shape

if broadcast == True:
    broadcast_shape = infer_broadcast_shape(arr, indices, axis)
    if broadcast_shape:
        index = torch.broadcast_to(index, broadcast_shape)
        src = torch.broadcast_to(src, broadcast_shape)
"""
        core = """
if reduce == 'assign':
    result = torch.scatter(input, dim, index, src)
else:
    result = torch.scatter_reduce(input, dim, index, src, reduce, include_self=include_self)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class PoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        head_code, map_code = self.apply_generic()
        func1 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride

def _get_same_padding_1d(input_size, kernel_size, stride):
    if stride is None:
        stride = kernel_size
    output_size = (input_size + stride - 1) // stride
    total_pad = max(0, (output_size - 1) * stride + kernel_size - input_size)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right

if isinstance(padding, str):
    if padding.upper() == "VALID":
        padding = 0
    elif padding.upper() == "SAME":
        input_size = x.shape[2]
        pad_left, pad_right = _get_same_padding_1d(input_size, kernel_size, stride)
        padding = pad_left # 对称填充
        if pad_left != pad_right:  # 非对称填充
            x = torch.nn.functional.pad(x, (pad_left, pad_right))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 1:  # [pad]
        padding = tuple(padding)
    elif len(padding) == 2:  # [pad_left, pad_right]
        pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        padding = 0
"""
        func2 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)
            
def _get_same_padding_2d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride, stride)
    output_size_h = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_w = (input_size[1] + stride[1] - 1) // stride[1]
    total_pad_h = max(0, (output_size_h - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_w = max(0, (output_size_w - 1) * stride[1] + kernel_size[1] - input_size[1])
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_h, pad_w

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3])
        pad_h, pad_w = _get_same_padding_2d(input_size, kernel_size, stride)
        padding = (pad_h[0], pad_w[0]) # 对称填充
        if pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1]))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 2: # [pad_height, pad_width]
        padding = tuple(padding)
    elif len(padding) == 4:
        if all(isinstance(p, (list, tuple)) for p in padding): # Paddle 的 4D 填充格式(NCHW 或 NHWC)
            if data_format == "NCHW":
                pad_top, pad_bottom = padding[2]
                pad_left, pad_right = padding[3]
            else:  # NHWC
                pad_top, pad_bottom = padding[1]
                pad_left, pad_right = padding[2]
        else: # [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]
            pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        padding = 0
"""
        func3 = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
if data_format == 'NDHWC':
    x = x.permute(0, 4, 1, 2, 3)
        
def _get_same_padding_3d(input_size, kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = (stride,) * 3
    output_size_d = (input_size[0] + stride[0] - 1) // stride[0]
    output_size_h = (input_size[1] + stride[1] - 1) // stride[1]
    output_size_w = (input_size[2] + stride[2] - 1) // stride[2]
    total_pad_d = max(0, (output_size_d - 1) * stride[0] + kernel_size[0] - input_size[0])
    total_pad_h = max(0, (output_size_h - 1) * stride[1] + kernel_size[1] - input_size[1])
    total_pad_w = max(0, (output_size_w - 1) * stride[2] + kernel_size[2] - input_size[2])
    pad_d = (total_pad_d // 2, total_pad_d - total_pad_d // 2)
    pad_h = (total_pad_h // 2, total_pad_h - total_pad_h // 2)
    pad_w = (total_pad_w // 2, total_pad_w - total_pad_w // 2)
    return pad_d, pad_h, pad_w

if isinstance(padding, str):
    if padding == "VALID":
        padding = 0
    elif padding == "SAME":
        input_size = (x.shape[2], x.shape[3], x.shape[4])  # (D, H, W)
        pad_d, pad_h, pad_w = _get_same_padding_3d(input_size, kernel_size, stride)
        padding = (pad_d[0], pad_h[0], pad_w[0]) # 对称填充
        if pad_d[0] != pad_d[1] or pad_h[0] != pad_h[1] or pad_w[0] != pad_w[1]: # 非对称填充
            x = torch.nn.functional.pad(x, (pad_w[0], pad_w[1], pad_h[0], pad_h[1], pad_d[0], pad_d[1]))
            padding = 0
elif isinstance(padding, (list, tuple)):
    if len(padding) == 3:  # [pad_depth, pad_height, pad_width]
        max_pad = [kernel_size[i] // 2 for i in range(3)]
        if any(p > m for p, m in zip(padding, max_pad)):
            pad_d, pad_h, pad_w = padding
            x = torch.nn.functional.pad(x, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d))
            padding = 0
        else:
            padding = tuple(padding)
    elif len(padding) == 6:  # [front, back, top, bottom, left, right]
        pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
    elif len(padding) == 5: # Paddle 的 5D 填充格式
        if data_format == "NCDHW":
            pad_front, pad_back = padding[2]
            pad_top, pad_bottom = padding[3]
            pad_left, pad_right = padding[4]
        else: # NDHWC
            pad_front, pad_back = padding[1]
            pad_top, pad_bottom = padding[2]
            pad_left, pad_right = padding[3]
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back))
        padding = 0
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        impl2 = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        impl3 = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if paddle_api.endswith("_pool1d"):
            code = head_code + func1.splitlines() + map_code + core.splitlines()
        elif paddle_api.endswith("_pool2d"):
            code = (
                head_code
                + func2.splitlines()
                + map_code
                + core.splitlines()
                + impl2.splitlines()
            )
        elif paddle_api.endswith("_pool3d"):
            code = (
                head_code
                + func3.splitlines()
                + map_code
                + core.splitlines()
                + impl3.splitlines()
            )
        else:
            return ConvertResult.error(
                paddle_api, f"Unsupported pooling api: {paddle_api}"
            )
        return ConvertResult.success(paddle_api, code)


# q
class QrRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
mode = locals().get('mode', 'reduced')
"""
        core = """
result = torch.linalg.qr(x, mode)
"""
        post = """
if mode == "r":
    result = result[1]
"""
        code = Code(
            preprocess=pre.splitlines(),
            core=core.splitlines(),
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(
            paddle_api, code, "result", is_torch_corresponding=False
        )


# r
class RankRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = f"result = torch.tensor(input.dim(),dtype=torch.int64)"
        code = Code(
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class Reduce_asRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
x_shape = list(x.shape)
t_shape = [1] * (x.dim() - target.dim()) + list(target.shape)
reduce_dims = [i for i, (xs, ts) in enumerate(zip(x_shape, t_shape)) if ts == 1 and xs != 1]
out = x.sum(dim=reduce_dims, keepdim=True)
result = out.view(target.shape)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ReshapeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
sh = []
if isinstance(shape,torch.Tensor):
    for i in shape:
        sh.append(i.item())
else:
    sh = shape
"""
        core = """
result = torch.reshape(x,sh)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class ReverseRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
dim = []
if isinstance(axis,int):
    dim.append(axis)
else:
    dim = axis
"""
        core = """
result = torch.flip(x,dim)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class Roi_aignRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import torchvision
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'output_size': 'output_size',
    'spatial_scale': 'spatial_scale',
    'sampling_ratio': 'sampling_ratio',
    'aligned': 'aligned'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
boxes = locals().get('boxes')
boxnum = locals().get('boxes_num')
ans = []
end = 0
"""
        core = """
for i in range(boxnum.shape[0]):
    begin = end
    end = end + int(boxnum[i])
    ans.append(boxes[begin:end,])
result = torchvision.ops.roi_align( **_kwargs, boxes = ans)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class Roi_poolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import torchvision
_kwargs = {}
for paddle_param, torch_param in {
    'x': 'input',
    'output_size': 'output_size',
    'spatial_scale': 'spatial_scale'
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]
    else:
        _kwargs[torch_param] = 1.0
boxes = locals().get('boxes')
boxnum = locals().get('boxes_num')
ans = []
end = 0
for i in range(boxnum.shape[0]):
    begin = end
    end = end + int(boxnum[i])
    ans.append(boxes[begin:end,])
"""
        core = f"result = {self.torch_api}(boxes = ans, **_kwargs)"
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code, "result")


class RollRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if 'shifts' in locals() and isinstance(shifts, torch.Tensor):
    if shifts.numel() == 1:
        shifts = shifts.item()
    else:
        shifts = shifts.tolist()
if 'axis' in locals() and isinstance(axis, torch.Tensor):
    if axis.numel() == 1:
        axis = axis.item()
    else:
        axis = axis.tolist()
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
        )
        return ConvertResult.success(paddle_api, code)


class ReduceRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, _ = self.apply_generic()
        pre = """
if isinstance(axis, (tuple, list)):
    tmp = []
    for a in axis:
        if torch.is_tensor(a):
            tmp.append(a.item())
        else:
            tmp.append(a)
    axis = tuple(tmp)
if torch.is_tensor(axis):
    if axis.dim() == 0:
        axis = axis.item()
    else:
        axis = tuple(axis.tolist())
"""
        if paddle_api == "paddle.mean":
            core = """
if axis is None:
    result = torch.mean(x)
else:
    result = torch.mean(x, dim=axis, keepdim=keepdim)
"""
            post = """
if axis is None and keepdim:
    result = result.view([1] * x.dim())
"""
        elif paddle_api == "paddle.prod":
            pre += """
if dtype is None:
    dtype = x.dtype
"""
            core = """
if axis is None:
    result = torch.prod(x, dtype = dtype)
elif isinstance(axis, int):
    result = torch.prod(x, dim=axis, keepdim=keepdim, dtype = dtype)
else:
    for a in axis:
        x = torch.prod(x, dim=a, keepdim=True, dtype=dtype)
    result = x
"""
            post = """
if isinstance(axis, tuple) and not keepdim:
    result = torch.squeeze(result, dim=axis)
"""
        elif paddle_api == "paddle.sum":
            core = f"result = torch.sum(x, dim=axis, keepdim=keepdim, dtype=dtype)"
            post = ""
        else:
            core = f"result = {self.torch_api}(x, dim=axis, keepdim=keepdim)"
            post = ""
        code = Code(
            preprocess=defaults_code + pre.splitlines(),
            core=core.splitlines(),
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


# s
class SampleNeighborsRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
eids = locals().get('eids', None)
sample_size = locals().get('sample_size', -1)
return_ids = locals().get('return_eids', False)
out_neighbors = []
out_count = []
out_eids = []
for node in input_nodes:
    start = colptr[node]
    end = colptr[node + 1]        
    neighbors = row[start:end]
    num_neighbors = neighbors.numel()
    edge_ids = torch.arange(start,end,dtype=torch.int64)

    if num_neighbors == 0:
        sampled = torch.tensor([], dtype=row.dtype)
        sampled_eids = torch.tensor([], dtype=torch.int64)
    elif sample_size == -1 or num_neighbors <= sample_size:
        sampled = neighbors
        sampled_eids = edge_ids
    else:
        sampled = neighbors[:sample_size]
        sampled_eids = edge_ids[:sample_size]

    out_neighbors.append(sampled)
    out_count.append(sampled.numel())
    out_eids.append(sampled_eids)

out_neighbors = torch.cat(out_neighbors) if out_neighbors else torch.tensor([], dtype=row.dtype)
out_count = torch.tensor(out_count, dtype=torch.int64)
if return_ids:
    out_eids = eids.index_select(0,torch.cat(out_eids)) if out_eids else torch.tensor([], dtype=eids.dtype)

if return_ids:
    result = (out_neighbors, out_count, out_eids)
else:
    result = (out_neighbors, out_count)

"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class SegmentMaxRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
num = int(segment_ids.max().item()) + 1
ans = torch.full((num,)+data.shape[1:], float('-inf'), dtype = data.dtype)
for idx in range(data.shape[0]): 
    seg_id = segment_ids[idx]
    val = data[idx]
    ans[seg_id][val > ans[seg_id]] = val[val > ans[seg_id]]
ans[ans == float('-inf')] = 0
result = ans
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ScatterRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
overwrite = locals().get('overwrite', True)
x = x.clone()
index = index.view(-1, 1)
try:
    updates = updates.expand_as(x)
except:
    pass
if not overwrite:
    for i in range(index.shape[0]):
        x[index[i]] = torch.zeros_like(x[index[i]])
    for i in range(index.shape[0]):
        x[index[i]] += updates[i]
else:
    for i in range(index.shape[0]):
        x[index[i]] = updates[i]
result = x    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ScatterndRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
output = torch.zeros(shape, dtype=updates.dtype).to(updates.device)
if index.numel() == 0:
    result = output + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        output[idx_tuple] += flat_updates[i]
    result = output    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class ScatterndaddRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
x = x.clone()
if index.numel() == 0:
    result = x + updates
else:
    flat_index = index.view(-1, index.size(-1))
    flat_updates = updates.reshape(flat_index.size(0), *updates.shape[index.dim()-1:])
    for i in range(flat_index.size(0)):
        idx_tuple = tuple(flat_index[i])
        x[idx_tuple] += flat_updates[i]
    result = x    
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class SeluRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        core = f"""
if scale == 1.0507009873554804934193349852946 and alpha == 1.6732632423543772848170429916717:
    result = {self.torch_api}(**_kwargs)
else:
    result = scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
"""
        code = Code(
            preprocess=defaults_code + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class SliceRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
for i,dim in enumerate(axes):
    if starts[i] < 0:
        starts[i] = starts[i] + input.shape[dim]
    if ends[i] < 0:
        ends[i] = ends[i] + input.shape[dim]
    ends[i] = min(ends[i],input.shape[dim])
    input = torch.narrow(input, dim, starts[i], ends[i]-starts[i])
result = input
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


class SplitRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
axis = locals().get("axis", 0)
if isinstance(axis, torch.Tensor):
    axis =axis.item()
if axis < 0:
    axis = len(x.shape) + axis
if not isinstance(num_or_sections, int):
    num = x.shape[axis]
    for i in num_or_sections:
        if i != -1:
            num = num - i
    for i in range(len(num_or_sections)):
        if num_or_sections[i] == -1:
            num_or_sections[i] = num
            break    
"""
        core = """
if isinstance(num_or_sections, int):
    result = torch.split(x, x.shape[axis] // num_or_sections, dim=axis)
else:
    result = torch.split(x, num_or_sections, dim=axis)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


class SqueezeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if 'axis' in locals():
    if isinstance(axis, torch.Tensor):
        if axis.numel() == 1:
            axis = axis.item()
        else:
            axis = tuple(axis.tolist())
    elif isinstance(axis, (list, tuple)):
        axis = tuple(axis)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
        )
        return ConvertResult.success(paddle_api, code)


class SortRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
axis = axis if axis >= 0 else x.dim() + axis
"""
        if self.torch_api.startswith("torch.Tensor"):
            core = "result, _ = x.sort(**_kwargs)"
        else:
            core = f"result, _ = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class SplitTensorRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
axis = axis if axis >= 0 else x.dim() + axis
if isinstance(num_or_sections, int):
    num_or_sections = x.shape[axis] // num_or_sections
elif isinstance(num_or_sections, list) and -1 in num_or_sections:
    num_or_sections[num_or_sections.index(-1)] = x.shape[axis] - sum(num_or_sections) - 1
"""
        core = "result = x.split(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class SlogdetRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = torch.linalg.slogdet(x)
"""
        post = """
result = torch.stack(result,0)
"""
        code = Code(core=core.splitlines(), postprocess=post.splitlines())
        return ConvertResult.success(paddle_api, code)


class SliceScatterRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
slices = [slice(None)] * x.dim()
for i, axis in enumerate(axes):
    slices[axis] = slice(starts[i], ends[i], strides[i])
shape = list(x.shape)
for i, axis in enumerate(axes):
    start, end, stride = starts[i], ends[i], strides[i]
    shape[axis] = (end - start + stride - 1) // stride
if list(value.shape) != shape:
    value = value.expand(shape)
result = x.clone()
result[tuple(slices)] = value
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class StandardGammaRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
rate = torch.ones_like(x)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post = "result = result.sample()"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=[post],
        )
        return ConvertResult.success(paddle_api, code)


class StanhRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = "x = x * scale_a"
        core = f"result = {self.torch_api}(**_kwargs)"
        post = "result = result * scale_b"
        code = Code(
            preprocess=defaults_code + [pre] + map_code,
            core=[core],
            postprocess=[post],
        )
        return ConvertResult.success(paddle_api, code)


class StridedSliceRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
shape = x.shape
index_list = [torch.arange(s) for s in shape]
for axis, start, end, stride in zip(axes, starts, ends, strides):
    dim_len = shape[axis]
    if start < 0:
        start += dim_len
    if end < 0:
        end += dim_len
    if stride > 0:
        start = min(max(start, 0), dim_len)
        end = min(max(end, 0), dim_len)
    else:
        start = min(max(start, -1), dim_len - 1)
        end = min(max(end, -1), dim_len - 1)
    index_list[axis] = torch.arange(start, end, step=stride)
grids = torch.meshgrid(*[ind if isinstance(ind, torch.Tensor) else torch.arange(shape[i]) 
                            for i, ind in enumerate(index_list)], indexing='ij')
result = x[grids]
"""
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ShardIndex(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
ignore_value = locals().get("ignore_value", -1)
shard_size = (index_num + nshards - 1) // nshards
lower = shard_id * shard_size
upper = (shard_id + 1) * shard_size

mask = (input >= lower) & (input < upper)
output = torch.full_like(input, ignore_value)

output[mask] = input[mask] - lower
result = output
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ScaleRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, _ = self.apply_generic()
        core = """
if bias_after_scale:
    result = scale * x + bias
else:
    result = scale * (x + bias)
if act is not None:
    if act == 'tanh':
        result = torch.tanh(result)
    elif act == 'sigmoid':
        result = torch.sigmoid(result)
    elif act == 'relu':
        result = torch.relu(result)
    elif act == 'softmax':
        result = torch.softmax(result, dim=-1)
"""
        code = Code(preprocess=defaults_code, core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ShapeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.tensor(input.shape)"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class SubtractRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if x.dtype == torch.bool:
    x = torch.tensor(x, dtype=y.dtype)
elif y.dtype == torch.bool:   
    y = torch.tensor(y, dtype=x.dtype)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class SwigluRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = locals().get("x")
y = locals().get("y", None)

if y == None:
    x, y = torch.chunk(x, 2, dim=-1)
"""
        core = "result = torch.nn.functional.silu(x) * y"
        code = Code(
            preprocess=pre.splitlines(),
            core=[core],
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class SegmentRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
num_segments = segment_ids.max().item() + 1
output_shape = (num_segments,) + data.shape[1:]
segment_ids = segment_ids.to(dtype=torch.int64)
"""
        core_max = """
result = torch.full(output_shape, float('-inf'), dtype=data.dtype)
result.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(data), data, 'amax')
result = torch.where(result == float('-inf'), torch.tensor(0.0, dtype=data.dtype), result)
"""
        core_min = """
result = torch.full(output_shape, float('inf'), dtype=data.dtype)
result.scatter_reduce_(0, segment_ids.unsqueeze(-1).expand_as(data), data, 'amin')
result = torch.where(result == float('inf'), torch.tensor(0.0, dtype=data.dtype), result)
"""
        core_sum = """
result = torch.zeros(output_shape, dtype=data.dtype)
result.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
"""
        core_mean = """
sum_result = torch.zeros(output_shape, dtype=data.dtype)
sum_result.scatter_add_(0, segment_ids.unsqueeze(-1).expand_as(data), data)
count = torch.zeros(num_segments, dtype=torch.int64)
count.scatter_add_(0, segment_ids, torch.ones_like(segment_ids, dtype=torch.int64))
count = count.view(num_segments, *[1] * (data.dim() - 1))
count = count.clamp(min=1)
result = sum_result / count.to(sum_result.dtype)
empty_mask = (count == 1) & (sum_result == 0)
result = torch.where(empty_mask, torch.tensor(0.0, dtype=result.dtype), result)
"""
        if paddle_api.endswith("max"):
            core = core_max
        elif paddle_api.endswith("min"):
            core = core_min
        elif paddle_api.endswith("sum"):
            core = core_sum
        elif paddle_api.endswith("mean"):
            core = core_mean
        else:
            return ConvertResult.error(
                paddle_api, f"Unsupported segment api: {paddle_api}"
            )
        code = Code(
            preprocess=pre.splitlines(),
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class SoftmaxMaskFuseRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = torch.softmax(x + mask, dim=-1)"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class SoftmaxMaskFuseUpperTriangleRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
batch, heads, seq_len, seq_len2 = x.shape
mask = torch.triu(torch.full((seq_len, seq_len2), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
mask = mask.view(1, 1, seq_len, seq_len2)
result = torch.softmax(x + mask, dim=-1)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# t
class TakeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if 'mode' not in locals():
    mode = 'raise'
def torch_take(x, index, mode='raise'):
    x_flat = x.reshape(-1)
    numel = x_flat.numel()
    if mode == 'raise':
        index_mask = (index >= 0) & (index < numel)
        valid_indices = torch.clamp(index, 0, numel - 1)  # 避免报错，先 clamp
        taken = torch.take(x_flat, valid_indices)
        taken[~index_mask] = 0.0  # 非法 index 位置手动填 0
        return taken.view(index.shape)
    elif mode == 'wrap':
        index_mod = ((index % numel) + numel) % numel
        result = torch.take(x_flat, index_mod)
    elif mode == 'clip':
        index_clipped = torch.clamp(index, 0, numel - 1)
        result = torch.take(x_flat, index_clipped)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return result.view(index.shape)
"""

        core = "result = torch_take(x, index, mode)"
        code = Code(preprocess=defaults_code + pre.splitlines() + map_code, core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class TriangularSolveRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
transpose = locals().get('transpose', False)
upper = locals().get('upper', True)
unitriangular = locals().get('unitriangular', False)
if transpose:
    x = x.transpose(-1,-2)
"""
        core = """
result = torch.linalg.solve_triangular(x,y,upper=upper,left=True,unitriangular=unitriangular)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


class TolistRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.tolist()"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code)


# u
class UnpoolRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre_1d = """
if data_format == "NLC":
    x = x.permute(0, 2, 1)        
"""
        pre_2d = """
if data_format == "NHWC":
    x = x.permute(0, 3, 1, 2)     
"""
        pre_3d = """
if data_format == "NDHWC":
    x = x.permute(0, 4, 1, 2, 3)      
"""
        pre = """
kernel_size = tuple(kernel_size) if isinstance(kernel_size, list) else kernel_size
stride = tuple(stride) if isinstance(stride, list) else stride
padding = tuple(padding) if isinstance(padding, list) else padding
output_size = list(output_size) if isinstance(output_size, tuple) else output_size
indices = indices.to(torch.int64)
"""
        core = f"result = {self.torch_api}(**_kwargs)"
        post_1d = """
if data_format == "NLC":
    result = result.permute(0, 2, 1)
"""
        post_2d = """
if data_format == "NHWC":
    result = result.permute(0, 2, 3, 1)
"""
        post_3d = """
if data_format == "NDHWC":
    result = result.permute(0, 2, 3, 4, 1)
"""
        if self.torch_api.endswith("1d"):
            pre = pre_1d + pre
            post = post_1d
        elif self.torch_api.endswith("2d"):
            pre = pre_2d + pre
            post = post_2d
        elif self.torch_api.endswith("3d"):
            pre = pre_3d + pre
            post = post_3d
        else:
            return ConvertResult.error(
                paddle_api, f"Unsupported unpool api: {paddle_api}"
            )
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=[core],
            postprocess=post.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)


class UnfoldRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        core = "result = x.unfold(**_kwargs)"
        code = Code(preprocess=defaults_code + map_code, core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


class UnsqueezeRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        defaults_code, map_code = self.apply_generic()
        pre = """
if 'axis' in locals():
    if isinstance(axis, torch.Tensor):
        if axis.numel() == 1:
            axis = axis.item()
        else:
            axis = axis.tolist()
    if isinstance(axis, tuple):
        axis = list(axis)
"""
        core = f"""
if isinstance(axis, list):
    input_tensor = x
    for ax in axis:
        input_tensor = torch.unsqueeze(input_tensor, ax)
    result = input_tensor
else:
    result = {self.torch_api}(**_kwargs)
"""
        code = Code(
            preprocess=defaults_code + pre.splitlines() + map_code,
            core=core.splitlines(),
        )
        return ConvertResult.success(paddle_api, code)

# v
class VecdotRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
x = locals().get('x')
y = locals().get('y')
axis = locals().get('axis', -1)
if torch.is_complex(x) or torch.is_complex(y):
    x = x.to(torch.complex128)
    y = y.to(torch.complex128)
elif x.dtype != y.dtype:
    if x.dtype == torch.float64 or y.dtype == torch.float64:
        target_dtype = torch.float64
    elif x.dtype == torch.float32 or y.dtype == torch.float32:
        target_dtype = torch.float32
    else:
        target_dtype = x.dtype
    x = x.to(target_dtype)
    y = y.to(target_dtype)
"""
        core = "result = torch.linalg.vecdot(x, y, dim=axis)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class ViewRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = """
result = x.view(shape_or_dtype)
"""
        code = Code(core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


class View_As_Rule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        core = "result = x.view_as(other)"
        code = Code(core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


class VariableLengthMemoryEfficientAttentionRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import math

def variable_length_memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    causal: bool = False,
    pre_cache_length: int = 0
) -> torch.Tensor:
    batch_size, num_heads, query_seq_len, head_size = query.shape
    key_seq_len = key.shape[2]
    # Broadcast key and value to match query's num_heads if needed
    if key.shape[1] != num_heads:
        # Repeat key and value along the num_heads dimension
        repeat_factor = num_heads // key.shape[1]
        key = key.repeat(1, repeat_factor, 1, 1)
        value = value.repeat(1, repeat_factor, 1, 1)
    # Default scale if not provided
    if scale is None:
        scale = math.sqrt(1.0 / head_size)
    scale = torch.tensor(scale, dtype=query.dtype, device=query.device)
    # Initialize mask if None
    if mask is None:
        mask = torch.zeros(batch_size, 1, query_seq_len, key_seq_len,
                        dtype=query.dtype, device=query.device)
    else:
        mask = mask[:, :, :query_seq_len, :key_seq_len]
    # Apply sequence length masking
    seq_mask = torch.ones(batch_size, 1, query_seq_len, key_seq_len,
                         dtype=torch.bool, device=query.device)
    for b in range(batch_size):
        q_len = seq_lens[b].squeeze().item()
        kv_len = kv_seq_lens[b].squeeze().item() + pre_cache_length
        seq_mask[b, :, q_len:, :] = False
        seq_mask[b, :, :, kv_len:] = False
    # Apply causal masking if enabled
    if causal:
        causal_mask = torch.tril(
            torch.ones(1, 1, query_seq_len, key_seq_len, dtype=torch.bool, device=query.device)
        )
        seq_mask = seq_mask & causal_mask
    # Compute attention scores: QK^T
    qk_res = torch.matmul(query, key.transpose(-1, -2))  # [batch_size, num_heads, query_seq_len, key_seq_len]
    # Apply scale
    attention = qk_res * scale
    attention = attention.masked_fill(~seq_mask, torch.finfo(attention.dtype).min)
    attention = attention + mask
    # Softmax over the last dimension
    softmax_result = torch.nn.functional.softmax(attention, dim=-1)
    softmax_result = softmax_result.masked_fill(~seq_mask, 0.0)
    # Compute output: softmax(QK^T)V
    result = torch.matmul(softmax_result, value)  # [batch_size, num_heads, query_seq_len, head_size]
    return result
"""
        core = "result = variable_length_memory_efficient_attention(**kwargs)"
        code = Code(preprocess=pre.splitlines(), core=[core])
        return ConvertResult.success(paddle_api, code, is_torch_corresponding=False)


# w
class WhereRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
_kwargs = {}
for paddle_param, torch_param in {
    'condition': 'condition',
    'x': 'input',
    'y': 'other',
}.items():
    if paddle_param in locals():
        _kwargs[torch_param] = locals()[paddle_param]    
if "input" in _kwargs and not isinstance(_kwargs['input'], torch.Tensor):
    _kwargs["input"] = torch.tensor(_kwargs['input'])
"""
        core = """
result = torch.where(**_kwargs)
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


# x


# y


# z
class ZerosRule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        pre = """
import re
dtype = locals().get("dtype", None)
if isinstance(shape,torch.Tensor):
    if shape.numel() == 1:
        shape = shape.item()
    else:
        li = []
        for i in shape:
            li.append(i.item())
        shape = li
if not dtype is None:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    else:
        if str(dtype).split('.')[0] in ["paddle", "numpy"]:
            dtype_str = str(dtype).split('.')[-1]
            dtype = getattr(torch, dtype_str)
        else:
            match = re.search(r"'(.+?)'", str(dtype))
            print(dtype)
            dtype_str = match.group(1)
            dtype_str = dtype_str.split('.')[-1]
            dtype = getattr(torch, dtype_str)
"""
        core = """
if dtype is None:
    result = torch.zeros(shape)
else:
    result = torch.zeros(shape, dtype=dtype)        
"""
        code = Code(preprocess=pre.splitlines(), core=core.splitlines())
        return ConvertResult.success(paddle_api, code)


# __
class __Pow__Rule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
tensor = locals().get('self')
other = locals().get('y')
result = tensor.__pow__(other)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


class __rshift__Rule(BaseRule):
    def apply(self, paddle_api: str) -> ConvertResult:
        impl = """
tensor = locals().get('x')
other = locals().get('y')
is_arithmetic = locals().get('is_arithmetic')
# setting default value for is_arithmetic
if is_arithmetic is None:
    is_arithmetic = True

def logical_right_shift(x: torch.Tensor, y: torch.Tensor):
    mask = (1 << (x.element_size() * 8 - 1)) - 1
    x_arithmetic, mask = x >> y, mask >> (y - 1)
    shifted = torch.where(y >= 1, x_arithmetic & mask, x)
    shifted = torch.where(y < 0, torch.zeros_like(x), shifted)
    return shifted
    
if is_arithmetic:
    result = tensor.__rshift__(other)
else:
    # logical right shift 
    result = logical_right_shift(tensor, other)
"""
        code = impl.splitlines()
        return ConvertResult.success(paddle_api, code)


__all__ = [  # type: ignore
    cls.__name__
    for cls in globals().values()
    if isinstance(cls, type) and issubclass(cls, BaseRule) and cls != BaseRule
]
