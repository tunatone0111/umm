import json


def run():
    captions = {}
    with open("data/datasets/flickr8k/Flickr8k.token.txt", "r") as f:
        lines = f.readlines()
        for i in range(len(lines) // 5):
            captions[lines[i * 5].split("#")[0]] = [
                line.split("#")[1][1:].strip() for line in lines[i * 5 : i * 5 + 5]
            ]

    with open("data/datasets/flickr8k/captions.json", "w") as fout:
        json.dump(captions, fout, indent=2)


if __name__ == "__main__":
    run()
