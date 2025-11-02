from datasets import load_dataset

from typing import Literal


def load_data(dname: Literal["datacomp_small", "hq_edit"]):
    if dname == "datacomp_small":
        ds = load_dataset("mlfoundations/datacomp_small", split="train")
    elif dname == "hq_edit":
        ds = load_dataset("UCSC-VLAA/HQ-Edit", split="train")
    else:
        raise ValueError(f"Dataset {dname} not found")

    return ds
