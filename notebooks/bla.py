import argparse
from typing import Optional

def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
parser.add_argument('--prompt', '--p', nargs='*', default=['x'])
parser.add_argument('--t', nargs='+', default=None)
args = parser.parse_args()

print(args.debug)
print(args.prompt)
print(args.t)

print('+'.join([]))