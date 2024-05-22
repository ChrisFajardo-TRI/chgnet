import torch
from pymatgen.core import Structure

from chgnet.model import CHGNet


# Replace torch.bincount as not supported by ONNX
def new_bincount(input_tensor, minlength: int = 0):
    counts = torch.tensor([], dtype=torch.int64)
    output_length = max(minlength, int(torch.max(input_tensor)) + 1)
    for value in range(output_length):
        value_count = (input_tensor == value).sum()
        new_count_tensor = torch.tensor([value_count])
        counts = torch.cat([counts, new_count_tensor])
    return counts

torch.bincount=new_bincount


# Normal model usage
structure = Structure.from_file("/Users/chrisfajardo/git_repos/pochi/Li.cif")
chgnet = CHGNet.load()
graph = chgnet.graph_converter(structure)
chgnet = chgnet.to("cpu")
chgnet = chgnet.eval()
print(chgnet.forward(graph))

# Trace model
traced = torch.jit.trace(chgnet, (graph,))

# Test save/load as Torchscript
traced.save("chgnet.torchscript")
c = torch.jit.load("chgnet.torchscript")
print(c.forward(graph))

# Save as ONNX
torch.onnx.export(traced, (graph,), f="chgnet.onnx", input_names =['graph'])
