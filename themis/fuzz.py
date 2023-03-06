import nnsmith
from nnsmith.materialize import Model
from nnsmith.backends.factory import BackendFactory 
from nnsmith.graph_gen import model_gen

from torch.fx import symbolic_trace
import torch
from torch import Tensor
import torch._dynamo
from nnsmith.narrow_spec import auto_opset
from nnsmith.abstract.op import *
from themis.render import Render
from nnsmith.logging import *

from typing import Tuple

import logging
import os
import ast
import traceback


ModelType = Model.init("torch", backend_target="cuda")
opset = auto_opset(ModelType)

LOG_DIR = "./report"
if os.path.exists(LOG_DIR):
    logging.warning(f"Report directory {LOG_DIR} already exists. Will delete it.")
    if os.path.isdir(LOG_DIR):
        os.rmdir(LOG_DIR)
    else:
        os.remove(LOG_DIR)
os.makedirs(LOG_DIR)

class BugReport(Exception):
    def __init__(self):
        super().__init__()

class Inconsistency(BugReport):
    def __init__(self, code, annotation="", target=None, output=None):
        super().__init__()
        self.code = code
        self.target = target
        self.output = output
        self.annotation = ""
    
    def report(self):
        return self._annotation
    
def verify_results(targets, outputs, render:Render):
    _ast = render.get_last_ast()
    _code = ast.unparse(_ast)
    if len(targets) != len(outputs):
        raise Inconsistency(_code, f"len(targets) != len(outputs) ({len(targets)} vs {len(outputs)})")
    for i, var in enumerate(render._model.output_map):
        target = targets[i]
        output = outputs[i]
        if not torch.allclose(target, output):
            raise Inconsistency(_code, f"Inconsistent result {var}", target, output)

while True:
    gen = model_gen(
        opset=opset,
        max_elem_per_tensor=65536,
        max_nodes=10,
        dtype_choices=['i32', 'bool', 'f32', 'i64'],
    )
    ir = gen.make_concrete()
    model = ModelType.from_gir(ir)
    model.refine_weights()  # either random generated or gradient-based.

    oracle = model.make_oracle()
    th_model = model.native_model 

    input_tensors = {k:torch.tensor(v) for k,v in oracle.input.items()}

    render = Render(th_model)
    mlist = th_model.mlist
    func = render.run()
    try:
        with torch.no_grad():
            target = func(mlist, **input_tensors)
        target = [t.detach().cpu() for t in target]
        with torch.no_grad():
            func_compiled = torch.compile(func)
            output = func_compiled(mlist, **input_tensors)
        output = [o.detach().cpu() for o in output]
        verify_results(target, output, render)
    except Inconsistency as e:
        logging.warning(f"[Inconsistency] {e.annotation}",)
        logging.warning(f"Expected: {e.target}")
        logging.warning(f"Output: {e.output}")
        logging.warning(f"code:\n{e.code}")
    except Exception as e:
        logging.warning(e)
        logging.warning(traceback.format_exc())

# symbolic_trace(th_model)

exit()

# model.native_model.forward(input_tensors)

traced_module = torch.jit.trace(model.native_model, input_tensors)

code = traced_module.code

print(code)

# new_forward_method = compile(code, '<string>', 'exec')
local = {}
