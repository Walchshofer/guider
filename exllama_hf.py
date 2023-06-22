import os
import sys
from pathlib import Path
from typing import *

sys.path.insert(0, str(Path("exllama")))
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig

import torch
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast

class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.generation_config = GenerationConfig()

        print(f'spawning with config: {config}')

        self.config.vocab_size = config.vocab_size

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        # TODO: May cause problem on multi-gpu inference?
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        # TODO: Some decoding methods (such as Contrastive Search) may not work at this time
        assert len(args) == 0, 'no *args should be passed to forward'
        use_cache = kwargs['use_cache']
        seq = kwargs['input_ids'][0].tolist()
        cache = kwargs['past_key_values'] if 'past_key_values' in kwargs else None
        if cache is None:
            cache = ExLlamaCache(self.ex_model)
            self.ex_model.forward(torch.tensor([seq[:-1]], dtype=torch.long), cache, preprocess_only=True)
        logits = self.ex_model.forward(torch.tensor([seq[-1:]], dtype=torch.long), cache).to(self.device)

        # return CausalLMOutputWithPast(logits=logits, past_key_values=cache if use_cache else None)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        assert len(model_args) == 0 and len(kwargs) == 0, "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
            
        # pretrained_model_name_or_path = Path(f'{shared.args.model_dir}') / Path(pretrained_model_name_or_path)
        print(f'Loading config from {pretrained_model_name_or_path}/config.json')
        config = ExLlamaConfig(pretrained_model_name_or_path / 'config.json')

        # from 'oobabooga/text-generation-webui/modules/exllama.py'
        weight_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert weight_path is not None, f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        
        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        config.act_order = True
        config.fused_attn = False
        config.fused_mlp_thd = 0

        # added by ansalern:
        config.vocab_size = 32001

        return ExllamaHF(config)