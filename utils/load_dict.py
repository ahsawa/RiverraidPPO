import torch


def load_dict(path):

    state_dict = torch.load(path)

    for key, value in state_dict.items():
        print(f"{key}: shape={value.shape}")

load_dict("Policies/policy_074233.pth")