import argparse
import os
import altered_dis

class ByteParser:
    def __init__(self):
        type = 'pyc'
        self.dataset = []

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
        pass

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
