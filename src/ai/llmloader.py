import getpass
import os
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.llms.fireworks import Fireworks
from pydantic_core.core_schema import no_info_after_validator_function

config = {}


def set_opts(cfg: dict):
    global config
    config = cfg.copy()

    global temp, ctx_len, max_tokens, n_gpu_layers
    temp = float(config["model_temp"]) if config["model_temp"] is not None else 0.75
    ctx_len = (
        int(config["model_ctx_len"]) if config["model_ctx_len"] is not None else 4096
    )
    max_tokens = (
        int(config["model_max_new_tokens"])
        if config["model_max_new_tokens"] is not None
        else 2048
    )
    n_gpu_layers = (
        int(config["model_gpu_layers"])
        if config["model_gpu_layers"] is not None
        else 24
    )
    return temp, ctx_len, max_tokens, n_gpu_layers


def load_local_llm():
    llm = LlamaCpp(
        model_path=config["model_path"],
        temperature=temp,
        n_gpu_layers=n_gpu_layers,
        n_ctx=ctx_len,
        max_tokens=max_tokens,
    )
    return llm


def load_fireworks_llm():
    if "FIREWORKS_API_KEY" not in os.environ:
        os.environ["FIREWORKS_API_KEY"] = getpass.getpass("Fireworks API Key:")

    # Initialize a Fireworks model
    llm = Fireworks(
        model="accounts/fireworks/models/zephyr-7b-beta",
        base_url="https://api.fireworks.ai/inference/v1/completions",
        max_tokens=max_tokens,
        temperature=temp,
    )
    return llm


def load_llm():
    if config["llm"] == "local":
        return load_local_llm()
    elif config["llm"] == "fireworks":
        return load_fireworks_llm()
    else:
        return load_local_llm()
