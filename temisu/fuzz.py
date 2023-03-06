import nnsmith
from nnsmith.materialize import Model
from nnsmith.backends.factory import BackendFactory 
from nnsmith.graph_gen import model_gen

from torch.fx import symbolic_trace
import torch
from torch import Tensor
import torch._dynamo

import numpy as np

from nnsmith.narrow_spec import auto_opset
from nnsmith.abstract.op import *
from .render import Render
from .logging import *

from typing import Tuple

import logging
import os
import ast
import traceback
import shutil

from nnsmith.util import (
    op_filter,
)

ModelType = Model.init("torch", backend_target="cpu")
ModelType.add_seed_setter()
opset = op_filter(
            auto_opset(ModelType, vulops=False),
        )

TEMISU_LOG.setLevel(logging.INFO)

LOG_DIR = "./report"
if os.path.exists(LOG_DIR):
    logging.warning(f"Report directory {LOG_DIR} already exists. Will delete it.")
    shutil.rmtree(LOG_DIR)
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
        self.annotation = annotation
    
    def report(self):
        return self.annotation

    def __str__(self):
        ret = self.annotation
        # ret += "\nCode:\n"
        # ret += self.code
        if self.target is not None:
            ret += f"\nExpected: {self.target}"
        if self.output is not None:
            ret += f"\nOutput: {self.output}"
        return ret
    
def verify_results(targets, outputs, render:Render):
    _ast = render.get_last_ast()
    _code = ast.unparse(_ast)
    if len(targets) != len(outputs):
        raise Inconsistency(_code, f"len(targets) != len(outputs) ({len(targets)} vs {len(outputs)})")
    for i, var in enumerate(render._model.output_map.keys()):
        target = targets[i]
        output = outputs[i]
        if not _no_nan_or_inf(target):
            continue
        if not np.allclose(target, output):
            # logging.warning(targets)
            raise Inconsistency(_code, f"Inconsistent result {var}", target, output)

def _no_nan_or_inf(target):
    return not np.any(np.isinf(target)) and not np.any(np.isnan(target))

tot_testcases = 0
pass_testcases = 0

while True:
    gen = model_gen(
        opset=opset,
        max_elem_per_tensor=65536,
        max_nodes=10,
        dtype_choices=['bool', 'f32', 'int32', 'int64'],
    )
    ir = gen.make_concrete()
    model = ModelType.from_gir(ir)
    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    tot_testcases += 1
    target = [oracle.output[k] for k in model.native_model.output_map]

    th_model = model.native_model 

    input_tensors = {k:torch.tensor(v) for k,v in oracle.input.items()}

    render = Render(th_model)
    mlist = th_model.mlist
    func = render.run()
    try:
        func_compiled = torch.compile(func, backend='inductor')
        with torch.no_grad():
            output = func_compiled(mlist, **input_tensors)
        if not isinstance(output, tuple):
            output = (output, )
        output = [o.detach().cpu().numpy() for o in output]
        verify_results(target, output, render)
        pass_testcases += 1
        TEMISU_LOG.info(f"[PASS] {pass_testcases}/{tot_testcases}")
    except Exception as e:
        TEMISU_LOG.warning(traceback.format_exc())
        
        # save report
        bug_id = tot_testcases - pass_testcases - 1
        report_dir = os.path.join(LOG_DIR, f"bug_{bug_id}")
        os.mkdir(report_dir)
        errlog_path = os.path.join(report_dir, "errlog.txt")

        with open(errlog_path, "w") as errlog_file:
            print(e, file=errlog_file)
        
        code_path = os.path.join(report_dir, "code.py")
        with open(code_path, "w") as code_file:
            print(render.get_full_code(), file=code_file)

        mlist_path = os.path.join(report_dir, "mlist.th")
        torch.save(render._model.mlist, mlist_path)

# symbolic_trace(th_model)

exit()

# model.native_model.forward(input_tensors)

traced_module = torch.jit.trace(model.native_model, input_tensors)

code = traced_module.code

print(code)

# new_forward_method = compile(code, '<string>', 'exec')
local = {}
