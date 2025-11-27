#!/usr/bin/env python3
"""Download CIFAR-10 and PTB datasets for Needle framework."""

import urllib.request
import os
import tarfile

def download_cifar10(data_dir="./data"):
    """Download and extract CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")

    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if os.path.isdir(cifar_dir):
        print(f"CIFAR-10 already exists at {cifar_dir}")
        return

    os.makedirs(data_dir, exist_ok=True)

    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_tar = os.path.join(data_dir, "cifar-10-python.tar.gz")

    print(f"Downloading from {cifar_url}...")
    urllib.request.urlretrieve(cifar_url, cifar_tar)

    print(f"Extracting to {data_dir}...")
    with tarfile.open(cifar_tar, 'r:gz') as tar:
        tar.extractall(data_dir)

    print(f"CIFAR-10 downloaded and extracted to {cifar_dir}")
    print(f"Removing tar file...")
    os.remove(cifar_tar)
    print("Done!")


def download_ptb(data_dir="./data"):
    """Download Penn Treebank dataset."""
    print("Downloading Penn Treebank dataset...")

    ptb_dir = os.path.join(data_dir, "ptb")
    os.makedirs(ptb_dir, exist_ok=True)

    ptb_base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
    files = ['train.txt', 'test.txt', 'valid.txt']

    for filename in files:
        filepath = os.path.join(ptb_dir, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists")
            continue

        url = ptb_base_url + filename
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)

    print(f"PTB downloaded to {ptb_dir}")
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for Needle framework")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory to store datasets (default: ./data)")
    parser.add_argument("--cifar10", action="store_true",
                        help="Download CIFAR-10 only")
    parser.add_argument("--ptb", action="store_true",
                        help="Download PTB only")

    args = parser.parse_args()

    # If no specific dataset requested, download both
    download_both = not (args.cifar10 or args.ptb)

    if args.cifar10 or download_both:
        download_cifar10(args.data_dir)
        print()

    if args.ptb or download_both:
        download_ptb(args.data_dir)

    print("\nAll datasets downloaded successfully!")
