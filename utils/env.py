from envs import ENV_DICT


def create_env(config: dict):
    key_params = "environment"
    key_construct = "environment_class"
    if key_params not in config or key_construct not in config:
        raise Exception(f"Both {key_params} and {key_construct} must in configuration dictionary.")

    EnvironmentClass = ENV_DICT[config[key_construct]]
    env = EnvironmentClass(**config[key_params])

    return env
