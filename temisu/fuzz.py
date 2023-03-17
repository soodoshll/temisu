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
from .ir import TFunction
from .logging import *
from .mutator import Mutator

from typing import Tuple

import logging
import os
import ast
import traceback
import shutil

from nnsmith.util import (
    op_filter,
)

# from tvm.relax.frontend.torch import relax_dynamo

device = 'cuda'

ModelType = Model.init("torch", backend_target=device)
ModelType.add_seed_setter()
opset = op_filter(
            auto_opset(ModelType, vulops=False),
            exclude=["core.LinearInterp", "core.BilinearInterp", "core.NearestInterp", "core.BicubicInterp",
             "core.TrilinearInterp", #"core.Ceil", "core.Floor", "core.PReLU", "core.Clip", "core.LeakyReLU",
             #"core.ReduceMax", "core.ReduceMin", "core.Abs", "core.Pad", "core.Atan", "core.ReduceMean", "core.Div"
             "core.Concat1", "core.Concat2", "core.Concat3", "core.Concat4", "core.Concat5", "core.Concat6",
             "core.Floor", "core.Round", "core.Ceil"
            ]
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
    def __init__(self, annotation="", target=None, output=None):
        super().__init__()
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
    
def verify_results(targets, outputs, tfunc, rtol=1e-05, atol=1e-08):
    if len(targets) != len(outputs):
        raise Inconsistency(f"len(targets) != len(outputs) ({len(targets)} vs {len(outputs)})")
    for i, var in enumerate(tfunc._model.output_map.keys()):
        target = targets[i]
        output = outputs[i]
        if not _no_nan_or_inf(target):
            continue
        if not np.allclose(target, output, rtol=rtol, atol=atol):
            # logging.warning(targets)
            raise Inconsistency(f"Inconsistent result {var}", target, output)

def _no_nan_or_inf(target):
    return not np.any(np.isinf(target)) and not np.any(np.isnan(target))

def _compile_and_run(prog, input_tensors, mode='default', backend='inductor'):
    compiled = torch.compile(prog, backend=backend, mode=mode)
    with torch.no_grad():
        output = compiled(mlist, **input_tensors)
    if not isinstance(output, tuple):
        output = (output, )
    output = [o.detach().cpu().numpy() for o in output]
    return output

def _save_report(report_dir="./report/tmp", tfunc=None, err=None):
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.makedirs(report_dir)
    if tfunc is not None:
        code_path = os.path.join(report_dir, "test.py")
        with open(code_path, "w") as code_file:
            print(tfunc.get_full_code(), file=code_file)
        mlist_path = os.path.join(report_dir, "param.th")
        torch.save((tfunc.patched_mlist(), input_tensors), mlist_path) 
    if err is not None:
        errlog_path = os.path.join(report_dir, "errlog.txt")
        with open(errlog_path, "w") as errlog_file:
            print(err, file=errlog_file)

tot_testcases = 0
pass_testcases = 0

while True:
    gen = model_gen(
        opset=opset,
        max_elem_per_tensor=65536,
        timeout_ms=10000,
        max_nodes=10,
        dtype_choices=['bool', 'f32', 'int32', 'int64', 'f64', ],
    )
    ir = gen.make_concrete()
    model = ModelType.from_gir(ir)
    model.refine_weights()  # either random generated or gradient-based.
    oracle = model.make_oracle()

    tot_testcases += 1
    target = [oracle.output[k] for k in model.native_model.output_map]

    th_model = model.native_model 

    input_tensors = {k:torch.tensor(v, device=device) for k,v in oracle.input.items()}

    tfunc = TFunction(th_model)
    th_model.to(device)
    mlist = th_model.mlist
    # backend = relax_dynamo() 
    backend = "inductor"

    try:
        # mutation_name = "origin"
        # output = _compile_and_run(func, input_tensors) 
        # verify_results(target, output, tfunc)
        for mutation_name, tfunc in Mutator(tfunc, input_tensors):
            TEMISU_LOG.info(f"mutate: {mutation_name}")
            if tfunc is None:
                continue
            func = tfunc.fn()
            _save_report(tfunc=tfunc)
            output = _compile_and_run(func, input_tensors, backend=backend)
            verify_results(target, output, tfunc, 1e-2, 1e-4)
            TEMISU_LOG.info(f"pass: {mutation_name}")
        pass_testcases += 1
        TEMISU_LOG.info(f"[PASS] {pass_testcases}/{tot_testcases}")
    except Exception as e:
        TEMISU_LOG.warning(traceback.format_exc())
        
        # save report
        report_dir = os.path.join(LOG_DIR, f"bug_{tot_testcases}_{mutation_name}")
        _save_report(report_dir, tfunc, e)

        TEMISU_LOG.info("failure report generated")