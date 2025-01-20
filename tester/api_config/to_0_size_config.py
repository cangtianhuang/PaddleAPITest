from config_analyzer import TensorConfig, APIConfig, analyse_configs
import copy

def is_0_size_tensor(tensor_config):
    for i in tensor_config.shape:
        if i == 0:
            return True
    return False

def is_0D_tensor(tensor_config):
    return len(tensor_config.shape) == 0

def get_tensor_configs(api_config):
    tensor_configs = []
    for arg_config in api_config.args:
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, list):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
        elif isinstance(arg_config, tuple):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])

    for key, arg_config in api_config.kwargs.items():
        if isinstance(arg_config, TensorConfig):
            tensor_configs.append(arg_config)
        elif isinstance(arg_config, list):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
        elif isinstance(arg_config, tuple):
            for j in range(len(arg_config)):
                if isinstance(arg_config[j], TensorConfig):
                    tensor_configs.append(arg_config[j])
    return tensor_configs

def to_0_size_config(api_config):
    result = []
    tensor_configs = get_tensor_configs(api_config)
    
    if len(tensor_configs) == 0:
        return []

    shape_len = len(tensor_configs[0].shape)
    shape_equal = True
    for tensor_config in tensor_configs:
        if is_0_size_tensor(tensor_config) or is_0D_tensor(tensor_config):
            return []
        if shape_len != len(tensor_config.shape):
            shape_equal = False

    for i in range(len(tensor_configs)):
        for j in range(len(tensor_configs[i].shape)):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            tmp_tensor_configs[i].shape[j] = 0
            result.append(tmp_api_config)

    if shape_equal:
        for j in range(shape_len):
            tmp_api_config = copy.deepcopy(api_config)
            tmp_tensor_configs = get_tensor_configs(tmp_api_config)
            for i in range(len(tensor_configs)):
                tmp_tensor_configs[i].shape[j] = 0
            result.append(tmp_api_config)
    return result

if __name__ == '__main__':
    config_0_size = []
    api_configs = analyse_configs("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config.txt")
    for api_config in api_configs:
        print(api_config.config)
        config_0_size = config_0_size + to_0_size_config(api_config)
    with open("/data/OtherRepo/PaddleAPITest/tester/api_config/api_config_0_size.txt", "w") as f:
        for api_config in config_0_size:
            f.write(str(api_config)+"\n")
        f.close()

        