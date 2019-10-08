import dis, marshal, sys

# Header size changed in 3.3. It might change again, but as of this writing, it hasn't.
header_size = 12 if sys.version_info >= (3, 3) else 8

with open(pycfile, "rb") as f:
    magic_and_timestamp = f.read(header_size)  # first 8 or 12 bytes are metadata
    code = marshal.load(f)                     # rest is a marshalled code object

dis.dis(code)
