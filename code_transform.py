import ast
import astor
import os
import argparse

curr_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('infile', type=argparse.FileType(), nargs="?", default="-")
parser.add_argument('--out', '-o', type=str, nargs="?", default="out.txt")
args = parser.parse_args()
with args.infile as infile:
    source = infile.read()
tree = ast.parse(source)
for fun in tree.body:
    if isinstance(fun, ast.FunctionDef):
        for _ag in fun.args.args:
            print(type(_ag.arg))
            fun.body.insert(0, ast.parse("print(type(" + str(_ag.arg) + "))").body[0])
            with open(args.outfile, 'w') as outfile:
                outfile.write(astor.to_source(fun))
