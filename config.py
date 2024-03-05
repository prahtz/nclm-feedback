from yacs.config import CfgNode as CN


class SystemArgs(CN):
    num_workers = 0


class DataArgs(CN):
    text_tokens_file = "data/tokenized/unique_text_tokens.k2symbols"
    manifest_dir = "data/tokenized/"
    max_duration = 120
    buffer_size = 40000
    shuffle_buffer_size = 100000
    bradley_terry_model = False


class ModelArgs(CN):
    valle_ckpt_path = "best-valid-loss.pt"
    ticodec_ckpt_path = "TiCodec/checkpoints/1codebook/g_00300000"
    ticodec_config_path = "TiCodec/egs/TiCodec-24k-320d/config_24k_320d_conv_1g1r_8g3k1s.json"


class RewardArgs(CN):
    model_name = "wer"


class PPOArgs(CN):
    batch_size = 256
    mini_batch_size = 1
    init_kl_coef = 0.2
    target = 6
    horizon = 10000
    train_steps = 1000
    lr = 1e-5
    whiten_rewards = False
    temperature = 1.0
    cliprange_value = 0.2
    do_dropout = True
    vf_coef = 0.1


class LogArgs(CN):
    exp_dir = "exp/wer_test"


class Parameters(CN):
    sys = SystemArgs()
    data = DataArgs()
    model = ModelArgs()
    reward = RewardArgs()
    ppo = PPOArgs()
    log = LogArgs()


def get_default_parameters():
    return Parameters()


if __name__ == "__main__":
    with open("default.yaml", "w") as f:
        f.write(get_default_parameters().dump())
