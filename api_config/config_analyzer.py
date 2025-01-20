import re
import collections
import paddle
import numpy
import math
import json
import torch
import paddle
import inspect

class TensorConfig:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.numpy_tensor = None
        self.paddle_tensor = None
        self.torch_tensor = None

    def __str__(self):
        return "TensorConfig("+str(self.shape)+","+self.dtype+")"
    def convert_dtype_to_torch_type(self, dtype):
        if dtype in ["float32", numpy.float32]:
            return torch.float32
        elif dtype in ['float16', numpy.float16]:
            return torch.float16
        elif dtype in ['float64', numpy.float64]:
            return torch.float64
        elif dtype in ['int16', numpy.int16]:
            return torch.int16
        elif dtype in ['int8', numpy.int8]:
            return torch.int8
        elif dtype in ['bool', numpy.bool]:
            return torch.bool
        elif dtype in ['bfloat16', numpy.uint16]:
            return torch.bfloat16
        elif dtype in ['uint8', numpy.uint8]:
            return torch.uint8
        elif dtype in ['int32', numpy.int32]:
            return torch.int32
        elif dtype in ['int64', numpy.int64]:
            return torch.int64
        elif dtype in ['complex64', numpy.complex64]:
            return torch.complex64
        elif dtype in ['complex128', numpy.complex128]:
            return torch.complex128
        else:
            raise ValueError(f'Unsupport dtype: {dtype}')

    def get_numpy_tensor(self):
        if self.numpy_tensor is None:
            dtype = "float32" if self.dtype == "bfloat16" else self.dtype
            self.numpy_tensor = numpy.random.random(self.shape).astype(dtype)
        return self.numpy_tensor
    
    def get_paddle_tensor(self):
        if self.paddle_tensor is None:
            self.paddle_tensor = paddle.to_tensor(
                self.get_numpy_tensor(),
                dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            )
            if self.dtype == "bfloat16":
                self.paddle_tensor = paddle.cast(self.paddle_tensor, dtype="uint16")
            self.paddle_tensor.stop_gradient = False
        return self.paddle_tensor
    def get_torch_tensor(self):
        device = torch.device("cuda:0")
        torch.set_default_device(device)
        if self.torch_tensor is None:
            self.torch_tensor = torch.tensor(
                self.get_numpy_tensor(),
                dtype=self.convert_dtype_to_torch_type(self.dtype)
                if self.dtype != 'bfloat16'
                else torch.float32,
                # requires_grad=True,
            )
            if self.dtype == "bfloat16":
                self.torch_tensor = self.torch_tensor.to(dtype=torch.bfloat16)
        return self.torch_tensor
        
class APIConfig:
    def __init__(self, config):
        config = config.replace("\n", "")
        self.config = config
        self.args = []
        self.kwargs = collections.OrderedDict()
        config = config.replace("Tensor", "TensorConfig")

        self.api_name, offset = self.get_api(config)

        while(True):
            tocken, offset = self.get_tocken(config, offset)
            if offset is None:
                return

            is_kwarg = config[offset] == '='
            if is_kwarg:
                key = tocken
                tocken, offset = self.get_tocken(config, offset+1)

            value, offset = self.get_one_arg(tocken, config, offset)
            
            if offset is None:
                return

            if is_kwarg:
                self.append_kwargs(key, value)
            else:
                self.append_args(value)

    def append_args(self, arg):
        self.args.append(arg)
        
    def append_kwargs(self, name, arg):
        self.kwargs[name] = arg
        
    def __str__(self):
        result = "APIConfig:"
        result = result + self.api_name + "("
        for arg in self.args:
            result = result + str(arg) + ","
        
        for key, value in self.kwargs.items():
            result = result + key + "=" + str(value) + ","
        
        result = result + ")"
        return result

    def get_tocken(self, config, offset):
        pattern = r'\b[A-Za-z0-9+-._]+\b|-[A-Za-z0-9+-._]+\b'
        match = re.search(pattern, config[offset:])
        if match:
            return match.group(), offset + match.start() + len(match.group())
        return None, None

    def get_api(self, config):
        return config[0:config.index("(")], len(config[0:config.index("(")])

    def get_tensor(self, config, offset):
        config = config[offset:]
        tensor_str = config[config.index("TensorConfig"):config.index(")")+1]
        return eval(tensor_str), offset + len(tensor_str)

    def get_dtype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.pir.core.convert_np_dtype_to_dtype_(tocken), offset

    def get_vartype(self, config, offset):
        tocken, offset = self.get_tocken(config, offset)
        return paddle.base.framework.convert_np_dtype_to_proto_type(tocken), offset

    def get_list(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "[":
                tmp = tmp + 1
            if config[i] == "]":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        list_str = config[offset: last_index+1]
        if "TensorConfig" not in list_str:
            list_str = list_str.replace(",", " ")

        offset = 1
        while(True):
            tocken, offset = self.get_tocken(list_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, list_str, offset)

            if offset is None:
                break

            result.append(value)

        return result, last_index+1

    def get_tuple(self, config, offset):
        result = []
        tmp = 0
        last_index = offset
        for i in range(offset, len(config)):
            if config[i] == "(":
                tmp = tmp + 1
            if config[i] == ")":
                tmp = tmp - 1
            if tmp == 0:
                last_index = i
                break
        
        tuple_str = config[offset: last_index+1]
        if "TensorConfig" not in tuple_str:
            tuple_str = tuple_str.replace(",", " ")

        offset = 1
        while(True):
            tocken, offset = self.get_tocken(tuple_str, offset)
            if offset is None:
                break

            value, offset = self.get_one_arg(tocken, tuple_str, offset)

            if offset is None:
                break

            result.append(value)

        return tuple(result), last_index+1

    def get_slice(self, config, offset):
        config = config[offset:]
        slice_str = config[config.index("("):config.index(")")+1]
        return eval("slice"+slice_str), offset+len(slice_str)

    def get_complex(self, config, offset):
        config = config[offset:]
        complex_str = config[config.index("("):config.index(")")+1]
        if "nan" in complex_str and complex_str[complex_str.index('nan')-1] != ".":
            complex_str = complex_str.replace("nan", "float('nan')")
        return eval("complex"+complex_str), offset+len(complex_str)

    def get_numpy_type(self, config, offset):
        config = config[offset:]
        numpy_type_str = config[config.index("(")+1:config.index(")")]
        return eval(numpy_type_str), offset+len(numpy_type_str)+2

    def get_one_arg(self, tocken, config, offset):
        if tocken == "TensorConfig":
            value, offset = self.get_tensor(config, offset-len(tocken))
        elif tocken == "Dtype":
            value, offset = self.get_dtype(config, offset)
        elif tocken == "VarType":
            value, offset = self.get_vartype(config, offset)
        elif tocken == "list":
            value, offset = self.get_list(config, offset)
        elif tocken == "tuple":
            value, offset = self.get_tuple(config, offset)
        elif tocken == "slice":
            value, offset = self.get_slice(config, offset)
        elif tocken == "complex":
            value, offset = self.get_complex(config, offset)
        elif tocken == "type":
            value, offset = self.get_numpy_type(config, offset)
        elif tocken == "nan":
            value = float('nan')
        elif config[offset] == '\"':
            value = tocken
        elif tocken is None:
            return None, None
        else:
            value = eval(tocken)
        return value, offset


def analyse_configs(config_path):
    with open(config_path, "r") as f:
        configs = f.readlines()
        f.close()

    api_configs = []
    for config in configs:
        api_config = APIConfig(config)
        api_configs.append(api_config)
    return api_configs
