import torch
import os 

def analyze_policy(path):

    state_dict = torch.load(path, map_location="cpu")

    total_params = sum(p.numel() for p in state_dict.values())

    size_bytes = os.path.getsize(path)

    size_mib = size_bytes / (1024 ** 2)
    size_mb  = size_bytes / (1000 ** 2)

    print(f"File: {path}")

    print(f"- Number of parameters: {total_params}")
    print(f"- Size of the file: {size_mib:.4f} MiB, {size_mb:.4f} MB")

    print(f"- Bytes/params: {size_bytes/total_params:.4f} [bytes/params]")
    print("\n")

    print("Architecture: ")

    for key, value in state_dict.items():
        print(f"{key}: shape={value.shape}")

    print("\n")

analyze_policy("Policies/policy_074233.pth")
analyze_policy("Policies/policy_112452.pth")
analyze_policy("Policies/policy_055004.pth")
analyze_policy("Policies/policy_082851.pth")

analyze_policy("Policies/policy_102215.pth")
analyze_policy("Policies/policy_134837.pth")
analyze_policy("Policies/policy_160004.pth")
analyze_policy("Policies/policy_baseline.pth")
