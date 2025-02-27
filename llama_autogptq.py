from guidance.llms import Transformers
from transformers import LlamaForCausalLM
from huggingface_hub import snapshot_download
import os
import glob
import sys
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

from model_roles import (
    Dolphin_Role,
    Llama2ChatRole,
    Llama2GuanacoRole,
    OpenChatRole,
    Vicuna1_3Role,
)

selected_role = None


def get_role():
    return selected_role
    # return Vicuna1_3Role
    # return Llama2ChatRole
    # return Llama2GuanacoRole
    # return Llama2UncensoredChatRole


class LLaMAAutoGPTQ(Transformers):
    """A HuggingFace transformers version of the LLaMA language model with Guidance support."""

    llm_name: str = "llama"

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):
        assert tokenizer is None, "We will not respect any tokenizer from the caller."
        assert isinstance(model, str), "Model should be a str with LLaMAAutoGPTQ"

        global selected_role
        if "guanaco" in model.lower():
            print("found a Guanaco model")
            selected_role = Llama2GuanacoRole
        elif "llama-2-7b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole
        elif "llama-2-13b-chat" in model.lower():
            print("found a llama2chat model")
            selected_role = Llama2ChatRole
        elif "vicuna" in model.lower():
            print("found a vicuna model")
            selected_role = Vicuna1_3Role
        elif "orca" in model.lower():
            print("found an orca model")
            selected_role = OpenChatRole
        elif "dolphin" in model.lower():
            print("found an orca model")
            selected_role = Dolphin_Role

        print(f"Initializing LLaMAAutoGPTQ with model {model}")

        branch = None
        if ":" in model:
            splits = model.split(":")
            model = splits[0]
            branch = splits[1]

        models_dir = "./models"
        name_suffix = model.split("/")[1]
        model_dir = f"{models_dir}/{name_suffix}"
        snapshot_download(repo_id=model, revision=branch, local_dir=model_dir)

        model_basename = find_safetensor_filename(model_dir)
        model_basename = model_basename.split(".safetensors")[0]

        print(f"found model with basename {model_basename} in dir {model_dir}")

        use_triton = False  # testing new autogptq
        low_vram_mode = "--low-vram" in sys.argv

        if low_vram_mode:
            print("low vram mode enabled")

        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        final_path = f"{model_dir}"

        model = AutoGPTQForCausalLM.from_quantized(
            # model,
            final_path,
            model_basename=model_basename,
            use_safetensors=True,
            inject_fused_mlp=low_vram_mode is False,
            inject_fused_attention=use_triton,
            device="cuda:0",
            use_triton=use_triton,
            warmup_triton=use_triton,
            quantize_config=None,
        )

        # model._update_model_kwargs_for_generation = (
        #     LlamaForCausalLM._update_model_kwargs_for_generation
        # )

        model.config.max_seq_len = 4096  # this is the one

        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

    @staticmethod
    def role_start(role):
        return get_role().role_start(role)

    @staticmethod
    def role_end(role):
        return get_role().role_end(role)


def find_safetensor_filename(dir):
    # Make sure the directory path ends with '/'
    if dir[-1] != "/":
        dir += "/"

    # Use glob to find all files with the given extension
    files = glob.glob(dir + "*.safetensors")

    # If there is at least one file, return the first one
    if len(files) == 0:
        print(f"Error: no safetensor file found in {dir}")
        return None
    elif len(files) == 1:
        return os.path.basename(files[0])
    else:
        print(f"Warning: multiple safetensor files found in {dir}, picking just one")
        return os.path.basename(files[0])
