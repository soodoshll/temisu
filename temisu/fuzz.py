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
from .render import render, Render
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
    
def verify_results(targets, outputs, tfunc):
    if len(targets) != len(outputs):
        raise Inconsistency(f"len(targets) != len(outputs) ({len(targets)} vs {len(outputs)})")
    for i, var in enumerate(tfunc._model.output_map.keys()):
        target = targets[i]
        output = outputs[i]
        if not _no_nan_or_inf(target):
            continue
        if not np.allclose(target, output):
            # logging.warning(targets)
            raise Inconsistency(f"Inconsistent result {var}", target, output)

def _no_nan_or_inf(target):
    return not np.any(np.isinf(target)) and not np.any(np.isnan(target))

tot_testcases = 0
pass_testcases = 0

tot_mutate = 0
pass_mutate = 0

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

    tfunc = render(th_model)
    func = tfunc.fn()
    mlist = th_model.mlist

    mutate_cnt = 0

    backend = 'inductor'

    try:
        func_compiled = torch.compile(func, backend=backend)
        with torch.no_grad():
            output = func_compiled(mlist, **input_tensors)
        if not isinstance(output, tuple):
            output = (output, )
        output = [o.detach().cpu().numpy() for o in output]
        verify_results(target, output, tfunc)
        pass_testcases += 1
        TEMISU_LOG.info(f"[PASS] {pass_testcases}/{tot_testcases}")
        mutator = Mutator(tfunc, input_tensors)

        tfunc = mutator._desolve_op()
        if tfunc is None:
            continue
        tot_mutate += 1
        mutate_cnt += 1
        func = tfunc.fn()
        func_compiled = torch.compile(func, backend=backend)
        with torch.no_grad():
            output = func_compiled(mlist, **input_tensors)
        if not isinstance(output, tuple):
            output = (output, )
        output = [o.detach().cpu().numpy() for o in output]
        verify_results(target, output, tfunc)
        pass_mutate += 1
        TEMISU_LOG.info(f"[PASS MUTATE] {pass_mutate}/{tot_mutate}")
    except Exception as e:
        TEMISU_LOG.warning(traceback.format_exc())
        
        # save report
        report_dir = os.path.join(LOG_DIR, f"bug_{tot_testcases}_{mutate_cnt}")
        os.mkdir(report_dir)
        errlog_path = os.path.join(report_dir, "errlog.txt")

        with open(errlog_path, "w") as errlog_file:
            print(e, file=errlog_file)
        
        code_path = os.path.join(report_dir, "test.py")
        with open(code_path, "w") as code_file:
            print(tfunc.get_full_code(), file=code_file)

        mlist_path = os.path.join(report_dir, "param.th")
        torch.save((tfunc.patched_mlist(), input_tensors), mlist_path)
