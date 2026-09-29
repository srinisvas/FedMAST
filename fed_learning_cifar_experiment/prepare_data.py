import os

from datasets import load_dataset

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cifar10_hf")
ds = load_dataset("uoft-cs/cifar10")
ds.save_to_disk(target)
print(f"Saved CIFAR-10 to {target}")
