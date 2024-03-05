from transformers import PretrainedConfig
from valle.models import VALLE
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from valle.data.collation import get_text_token_collater
from typing import Dict
from transformers import PreTrainedTokenizer, BatchEncoding
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from valle.models.macros import NUM_AUDIO_TOKENS
from config import Parameters
from pathlib import Path


class VALLETokenizer(PreTrainedTokenizer):
    def __init__(self):
        super().__init__()

    def get_vocab(self):
        return {}

    def pad(self, x, **kwargs):
        x = {
            "input_ids": pad_sequence([x[i]["input_ids"] for i in range(len(x))], batch_first=True, padding_value=NUM_AUDIO_TOKENS),
            "attention_mask": pad_sequence([x[i]["attention_mask"] for i in range(len(x))], batch_first=True, padding_value=0),
        }
        return BatchEncoding(data=x)


class VALLEConfig(PretrainedConfig):
    model_type = "valle"

    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 12,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = False,
        num_quantizers: int = 1,
        do_dropout: bool = True,
        temperature: float = 1.0,
        **kwargs,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.norm_first = norm_first
        self.add_prenet = add_prenet
        self.prefix_mode = prefix_mode
        self.share_embedding = share_embedding
        self.nar_scale_factor = nar_scale_factor
        self.prepend_bos = prepend_bos
        self.num_quantizers = num_quantizers
        self.hidden_size = d_model
        self.do_dropout = do_dropout
        self.temperature = temperature
        super().__init__(**kwargs)


class VALLEWrapper(PreTrainedModel):
    config_class = VALLEConfig

    def __init__(self, config: VALLEConfig):
        super().__init__(config)
        self.model = VALLE(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            norm_first=config.norm_first,
            add_prenet=config.add_prenet,
            prefix_mode=config.prefix_mode,
            share_embedding=config.share_embedding,
            num_quantizers=config.num_quantizers,
            do_dropout=config.do_dropout,
        )
        self.lm_head = self.model.ar_predict_layer
        self.decoder = self.model.ar_decoder
        self.text_token_collater = get_text_token_collater("valle/examples/libritts/data/tokenized/unique_text_tokens.k2symbols")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        # Unwrapping x, x_lens, y, y_lens from the provided input_ids.
        text_eos_idx = self.text_token_collater.token2idx[self.text_token_collater.eos_symbol]
        x_lens = [torch.nonzero(input_ids[i] == text_eos_idx)[0] + 1 for i in range(input_ids.shape[0])]
        text_sep_idx = max(x_lens)
        x = input_ids[:, :text_sep_idx]
        x_lens = torch.stack(x_lens).squeeze(-1)

        y = input_ids[:, text_sep_idx:]
        y_lens = [
            torch.nonzero(y[i] == NUM_AUDIO_TOKENS)[0] + 1 if y[i][-1] == NUM_AUDIO_TOKENS else torch.tensor([y.shape[1]], device=y.device)
            for i in range(input_ids.shape[0])
        ]
        y_lens = torch.stack(y_lens).squeeze(-1)
        y = y[:, : y_lens.max().item()].unsqueeze(-1)
        # y[y == NUM_AUDIO_TOKENS] = 0

        # Forward to VALLE
        _, loss, _, logits, hidden_states = self.model.forward(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            reduction="sum",
            train_stage=1,
            return_logits=True,
            output_hidden_states=True,
        )
        # Append dummy tensors to logits and hidden states to match the expected output size of TRL
        # NOTE: dummy text tokens are part of queries and are ignored during the PPO loss computation,
        #       while dummy audio tokens are excluded by the attention mask. TODO: test the latter.
        logits = torch.cat(
            [
                torch.zeros((logits.shape[0], text_sep_idx, logits.shape[2]), device=logits.device),
                logits,
                torch.zeros(
                    (logits.shape[0], input_ids.shape[1] - y_lens.max().item() - x_lens.max().item(), logits.shape[2]), device=logits.device
                ),
            ],
            dim=1,
        )
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        hidden_states = torch.cat(
            [
                torch.zeros((hidden_states.shape[0], text_sep_idx, hidden_states.shape[2]), device=hidden_states.device),
                hidden_states,
                torch.zeros(
                    (hidden_states.shape[0], input_ids.shape[1] - y_lens.max().item() - x_lens.max().item(), hidden_states.shape[2]),
                    device=hidden_states.device,
                ),
            ],
            dim=1,
        )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=(hidden_states,),
        )

    def generate(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        top_k: int = -100,
    ):
        response, response_lens = self.model.batched_inference(
            x=x,
            x_lens=x_lens,
            y=y,
            y_lens=y_lens,
            enroll_x_lens=None,
            top_k=top_k,
            temperature=self.config.temperature,
        )
        return response, response_lens


def register_valle():
    AutoConfig.register("valle", VALLEConfig)
    AutoModel.register(VALLEConfig, VALLEWrapper)
    AutoModelForCausalLM.register(VALLEConfig, VALLEWrapper)


def wrap_and_save_valle_checkpoint(params: Parameters, save_dir: str = "valle"):
    checkpoint = torch.load(params.model.valle_ckpt_path, map_location="cpu")
    model_name = str(checkpoint["exp_dir"]).split("/")[-1]
    model_path = str(Path(save_dir) / model_name)
    config = VALLEConfig(
        d_model=checkpoint["decoder_dim"],
        nhead=checkpoint["nhead"],
        num_layers=checkpoint["num_decoder_layers"],
        add_prenet=checkpoint["add_prenet"],
        prefix_mode=checkpoint["prefix_mode"],
        share_embedding=checkpoint["share_embedding"],
        nar_scale_factor=checkpoint["scale_factor"],
        prepend_bos=checkpoint["prepend_bos"],
        num_quantizers=checkpoint["num_quantizers"],
        do_dropout=params.ppo.do_dropout,
        temperature=params.ppo.temperature,
    )

    model = VALLEWrapper(config)
    model.model.load_state_dict(checkpoint["model"])
    model.save_pretrained(model_path)
    register_valle()
    return model_path


def load_valle_with_value_head(model_path: str = "valle") -> AutoModelForCausalLMWithValueHead:
    register_valle()
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    return model
