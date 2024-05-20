import numpy as np
import torch
from pymatgen.core import Structure

from chgnet.model import CHGNet

# If the above line fails in Google Colab due to numpy version issue,
# please restart the runtime, and the problem will be solved

np.set_printoptions(precision=4, suppress=True)

def new_bincount(input_tensor, minlength: int = 0):
    counts = torch.tensor([], dtype=torch.int64)
    output_length = max(minlength, int(torch.max(input_tensor)) + 1)
    for value in range(output_length):
        value_count = (input_tensor == value).sum()
        new_count_tensor = torch.tensor([value_count])
        counts = torch.cat([counts, new_count_tensor])
    return counts

torch.bincount=new_bincount


structure = Structure.from_file("/Users/chrisfajardo/git_repos/pochi/Li.cif")

chgnet = CHGNet.load()
graph = list(chgnet.graph_converter(structure))
chgnet = chgnet.to("cpu")
chgnet = chgnet.eval()
print(chgnet.forward([graph]))

traced = torch.jit.trace(chgnet, ([graph],))
traced.save("chgnet.torchscript")

c = torch.jit.load("chgnet.torchscript")
c.forward([graph])

torch.onnx.export(traced, [graph], f="chgnet.onnx")