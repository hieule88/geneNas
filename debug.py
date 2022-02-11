import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--baseline", action='store_true')

args = parser.parse_args()

print(args.baseline)

