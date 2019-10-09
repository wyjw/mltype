import argparse
import os
import altered_dis
import torch

class ByteParser:
    def __init__(self):
        type = 'pyc'
        self.dataset = []
        self.function_split = {}

    def parse_instrs(self, list_of_instrs):
        #for i in list_of_instrs:
        for i in list_of_instrs:
            temp_dict = {}
            temp_dict['opname'] = i.opname
            temp_dict['opcode'] = i.opcode
            temp_dict['arg'] = i.arg
            temp_dict['argval'] = i.argval
            temp_dict['argrepr'] = i.argrepr
            temp_dict['offset'] = i.offset
            temp_dict['starts_line'] = i.starts_line
            temp_dict['is_jump_target'] = i.is_jump_target
            self.dataset.append(temp_dict)

    def breakdown_by_func(self):
        if len(self.dataset) == 0:
            pass
        fun = ''
        for i in self.dataset:
            if 'code object' in i['argrepr']:
                fun = i['argrepr'].split('code object')[1].split(' ')[1]
            if fun not in self.function_split:
                self.function_split[fun] = []
            self.function_split[fun].append(i['opname'])
        print(self.function_split)


if __name__ == "__main__":
    curr_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=argparse.FileType(), nargs='?', default='-')
    parser.add_argument('--input_dir', default = curr_dir + 'test_programs',
                        type = str, help = "Directory of all the python bytecode.")
    args = parser.parse_args()
    with args.infile as infile:
        source = infile.read()
    code = compile(source, args.infile.name, "exec")

    byteP = ByteParser()
    byteP.parse_instrs(list(altered_dis.get_instructions(code)))
    byteP.breakdown_by_func()

result = subprocess.run(['python', args.infile + ".out"], stdout = subprocess.PIPE)
output = result.stdout
