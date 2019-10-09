import ast
import astor
import os
import argparse
import subprocess

curr_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str, nargs="?", default="-")
#parser.add_argument('--out', '-o', type=str, nargs="?", default="test_programs/queen.py.out")
args = parser.parse_args()
with open(args.infile, 'r') as infile:
    source = infile.read()
tree = ast.parse(source)
with open(str(args.infile + ".out"), 'w') as outfile:
    for fun in tree.body:
        print(astor.to_source(fun))
        if isinstance(fun, ast.FunctionDef):
            for _ag in fun.args.args:
                fun.body.insert(0, ast.parse('print("' + str(fun.name) + '")').body[0])
                fun.body.insert(0, ast.parse("print(type(" + str(_ag.arg) + "))").body[0])
                outfile.write(astor.to_source(fun))
        else:
            outfile.write(astor.to_source(fun))
