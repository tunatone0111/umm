import torch
import tqdm
import json
from .models.bagel_chat_pipeline import BagelChatPipeline
from datasets import load_dataset


def run():
    pipe = BagelChatPipeline.from_pretrained(
        model_path="data/models/BAGEL-7B-MoT",  # Update this path
        device="cuda",
        max_memory_per_gpu="80GiB",
    )

    ds = load_dataset("lmms-lab/POPE", split="test")

    correct = 0  # Track number of correct predictions

    with tqdm.tqdm(total=len(ds)) as pbar:
        for i, row in enumerate(ds, 1):
            with torch.inference_mode():
                # Standard mode
                result = pipe(
                    images=[row["image"]],
                    prompt=row["question"],
                    # + "\nAnswer the question using a single word or phrase.",
                    cot=True,
                    return_full_reasoning=True,
                )
                pred = "yes" if "yes" in result["answer"].lower() else "no"
                # Count correct answers for cumulative accuracy
                if pred == row["answer"].lower():
                    correct += 1

                result = {
                    "category": row["category"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prediction": pred,
                    "reasoning": result["reasoning"],
                }
                pbar.update(1)
                pbar.set_description(f"Pred: {pred} | Acc: {correct / i:.4f}")

            with open("results_cot.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    run()
