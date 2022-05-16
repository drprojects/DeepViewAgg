KNOWN_MODULES = {
    'ADE20KResNet18PPM': 512,
    'ADE20KResNet18TruncatedLayer4': 512,
}

def extract_output_nc(model_config):
    """ Extracts the number of channels at the output of the network form the model config
    """
    if model_config.get('output_nc') is not None:
        output_nc = model_config.output_nc
    elif model_config.get('up_conv') is not None:
        output_nc = model_config.up_conv.up_conv_nn[-1][-1]
    elif model_config.get('innermost') is not None:
        output_nc = model_config.innermost.nn[-1]
    elif model_config.down_conv.module_name in KNOWN_MODULES.keys():
        output_nc = KNOWN_MODULES[model_config.down_conv.module_name]
    else:
        raise ValueError("Input model_config does not match expected pattern")
    return output_nc
