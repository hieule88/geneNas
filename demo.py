import argparse

from transformers import AutoTokenizer

# import pytorch_lightning as pl
import torch

from problem import BestModel

labels = ["Negative", "Neutral", "Positive"]


def pipeline(text):
    global tokenizer, model
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    label_idx = torch.argmax(output[1]).item()
    return f"The sentence is {labels[label_idx]}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("GeneNAS demo")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = BestModel.load_from_checkpoint(args.model_path)

    text = input("Input sentence: ")
    print(pipeline(text))
