import torch
import random
from functools import cache
from enum import Enum
import numpy as np

from nnsmith.abstract.op import *

from .ir import *

ScalarizeMethod = Enum('ScalarizeMethod', ['ELEMENT', 'SHAPE', 'MAX', 'MIN', 'SUM', 'MEAN'])

def _scalarize(k:str, v:torch.Tensor):
    if len(v.shape) == 0:
        return Variable(k), v
    p = random.choice(list(ScalarizeMethod))
    # p = ScalarizeMethod.ELEMENT
    if p == ScalarizeMethod.SHAPE:
        dim = random.randint(0, len(v.shape) - 1)
        k = Variable(k, suffix=f".shape[{dim}]")
        v = v.shape[dim]
    elif p == ScalarizeMethod.ELEMENT:
        dim = Mutator._choose_pos(v)
        k = Variable(k, idx=list(map(str, dim)))
        v = v[dim]
    elif p == ScalarizeMethod.MIN:
        k  = Variable(k, suffix=".min()")
        v = v.min()
    elif p == ScalarizeMethod.MAX:
        k  = Variable(k, suffix=".max()")
        v = v.max()
    elif p == ScalarizeMethod.SUM:
        k  = Variable(k, suffix=".sum()")
        v = v.sum()
    elif p == ScalarizeMethod.MEAN:
        if not torch.is_floating_point(v):
            k = Variable(k, suffix=".float().mean()")
            v = v.float().mean()
        else:
            k = Variable(k, suffix=".mean()")
            v = v.mean()

    return k, v

def _generate_true_condition(tensor_map):
    (k1, v1), (k2, v2) = random.sample(tensor_map.items(), 2)

    k1, v1 = _scalarize(k1, v1)
    k2, v2 = _scalarize(k2, v2)
    
    op = '<' if v1 < v2 else '>='
    expr = BinaryCompExpr(k1, k2, op) 
    return expr

class Mutator(object):
    def __init__(self, tfunc, inputs):
        self._tfunc : TFunction = tfunc
        self._model = tfunc._model
        self._inputs = inputs
        self._mlist = tfunc._model.mlist

        self._inst_list : List[Instruction] = tfunc.get_inst_list()
        self._cached_profile = None

        self._x = None
    
    def _init_locals(self):
        local_dict = {}
        
        local_dict['mlist'] = self._mlist
        for k,v in self._inputs.items():
            local_dict[k] = v

        return local_dict

    # Be cautious about cache!
    def profile(self, step=-1):
        local_dict = self._init_locals()
        for idx, stmt in enumerate(self._inst_list):
            if idx == step:
                break
            module = ast.Module([stmt.to_ast()], [], lineno=0, col_offset=0)
            ast.fix_missing_locations(module)
            with torch.no_grad():
                exec(compile(module, "<string>", "exec"), globals(), local_dict)
        return {k : v for k, v in local_dict.items() if isinstance(v, torch.Tensor)}

    def insert_tcb(self):
        n_stmts = len(self._inst_list)
        start = random.randint(0, n_stmts - 1)
        tensor_table = self.profile(start)

        if len(tensor_table) < 2:
            return self._tfunc

        cond = _generate_true_condition(tensor_table)
        end = random.randint(start + 1, n_stmts)

        if_inst = IfInstruction(cond, self._inst_list[start:end], )
        del self._inst_list[start:end]
        self._inst_list.insert(start, if_inst)

        return self._tfunc
        
    @staticmethod
    def _choose_pos(t):
        return tuple([random.randint(0, d - 1) for d in t.shape])
    
    def desolve_op(self):
        candidates = []
        for idx, inst in enumerate(self._inst_list):
            if isinstance(inst, CoreInstruction):
                if isinstance(inst._op, ElementWiseUnaryOp):
                    candidates.append(idx) 
        chosen = random.choice(candidates)
        inst = self._inst_list[chosen]

        snapshot = self.profile(chosen + 1)
        out = snapshot[inst._outs[0].name()]

        if len(out.shape) == 0:
            return self._tfunc

        dim = random.randint(0, len(out.shape) - 1)
        init = InitInstruction(inst._outs[0].name(), out.shape, out.dtype, out.device)
        loop = ElementwiseLoop(inst, dim, out.shape)

        self._inst_list[chosen] = loop
        self._inst_list.insert(chosen, init)
        return self._tfunc

    def origin(self):
        return self._tfunc

    def _insert_input_to_profile(self, profile):
        tensor_map = profile.copy()
        for k, v in self._inputs.items():
            tensor_map[k] = v        
        return tensor_map

    def matmul_then_inverse(self):
        profile = self.profile()
        candidates = []
        for i, snapshot in enumerate(profile[:-1]):
            for var, val in snapshot.items():
                if val.dtype in [torch.float32, torch.float64] and len(val.shape) >= 2:
                    candidates.append((i, var, val))
        
        if len(candidates) == 0:
            return self._tfunc

        inst_list = self._inst_list 
        inst_no, var, val = random.choice(candidates)
        dim = val.shape[-1]

        a = torch.rand([dim, dim], dtype=val.dtype, device=val.device)
        a_module = ConstFn(a)
        self._mlist.append(a_module)
        init_stmt = CoreInstruction(self._model, a_module, [], ['tmp'], Constant(a.shape))
        inst_list.insert(inst_no, init_stmt)
        matmul_stmt = CoreInstruction(self._model, torch.matmul, [var, 'tmp'], [var], MatMul())
        inst_list.insert(inst_no + 1, matmul_stmt)

        # self._clean_cache()

        dep = None
        for i in range(inst_no + 2, len(inst_list)):
            if var in inst_list[i].inputs():
                dep = i
                break
        if var in self._tfunc._model.output_map:
            dep = len(inst_list)
        if dep is not None:
            a_inv = torch.inverse(a.cuda())
            a_inv = a_inv.to(a.device)
            a_inv_module = ConstFn(a_inv)
            self._mlist.append(a_inv_module)
            inv_init_stmt = CoreInstruction(self._model, a_inv_module, [], ['tmp_inv'], Constant(a.shape))
            inv_stmt = CoreInstruction(self._model, torch.matmul, [var, 'tmp_inv'], [var], MatMul())
            inst_list.insert(dep, inv_init_stmt)
            inst_list.insert(dep + 1, inv_stmt)

        self._clean_cache()
        return self._tfunc


    def modify_then_recover(self):
        chosen = random.randint(0, len(self._inst_list) - 1)
        snapshot = self.profile(chosen)

        candidates = {k : v for k, v in snapshot.items() if len(v.shape) > 0}

        data_ptr = set([c().data.untyped_storage().data_ptr() for c in self._model.mlist if c.__class__.__name__ == 'ConstFn'])
        data_ptr.update([v.untyped_storage().data_ptr() for k, v in self._inputs.items()])

        for k, v in snapshot.items():
            if k in candidates and v.untyped_storage().data_ptr() in data_ptr:
                # print("remove const fn")
                del candidates[k]
        if len(candidates) == 0:
            return self._tfunc
        var, val = random.choice(list(candidates.items()))
        pos = self._choose_pos(val)

        alias = set()
        # find alias
        for k, v in snapshot.items():
            if v.untyped_storage().data_ptr() == val.untyped_storage().data_ptr():
                alias.add(k)

        var_element = Variable(var, list(map(str, pos)))

        backup_inst = SimpleAssignInstruction("backup", var_element)
        modify_inst = SimpleAssignInstruction(var_element, str(torch.rand([]).to(val.dtype).numpy()))
        recover_inst = SimpleAssignInstruction(var_element, "backup")

        inst_list = self._inst_list
        inst_list.insert(chosen, backup_inst)
        inst_list.insert(chosen + 1, modify_inst)
        
        dep = None
        for i in range(chosen + 2, len(inst_list)):
            if len(alias.intersection(inst_list[i].inputs())) > 0:
                dep = i
                break
        if dep is None and len(alias.intersection(self._model.output_map)) > 0:
            dep = len(inst_list)

        if dep is not None:
            inst_list.insert(dep, recover_inst)
        
        return self._tfunc

    transforms = ["origin", 
                  # "matmul_then_inverse",
                  "modify_then_recover",
                  "desolve_op",
                  "insert_tcb", 
                  ]

    def __iter__(self):
        self._ptr = 0
        return self
    
    def __next__(self):
        if self._ptr >= len(self.transforms):
            raise StopIteration
        transform = self.transforms[self._ptr]
        transform_fn = getattr(self, transform)
        tfunc = transform_fn()
        self._ptr += 1
        return transform, tfunc

ELEMENTWISE_OP = [
    ReLU, Add, GELU, LeakyReLU, Sigmoid, Sin, Cos, Asin, Acos, Tan, Atan,
    Abs, Mul, Div, Max, Min, Equal, Greater, Less, And, Or, Xor, Pow, Floor,
    Ceil, Clip, Round, Sqrt, Log2, Neg
]

class OpDesolver(object):
    def __init__(self, tfunc:TFunction, tensor_map):
        self._tfunc = tfunc
        self._model = tfunc._model
        self._inst_list = tfunc._inst_list
        self._tensor_map = tensor_map

    def _shape(self, t):
        return self._tensor_map[t].shape

    def _check_shape(self, inps, out):
        out_shape = self._shape(out)
        if len(out_shape) == 0:
            return False
        for inp in inps:
            if self._shape(inp) != out_shape:
                return False
        return True

    def run(self):
        instructions = self._inst_list
        candidates = []
        for stmt_idx, mlist in enumerate(instructions):
            _, inps, outs, op = mlist.get()
            if len(outs) != 1:
                continue
            # if isinstance(op, MatMul):
                # candidates.append(stmt_idx)
                # continue
            if not self._check_shape(inps, outs[0]):
                continue
            for op_type in ELEMENTWISE_OP:
                if isinstance(op, op_type):
                    candidates.append(stmt_idx)
                    break
        
        if len(candidates) == 0:
            return None

        chosen = random.choice(candidates)
        _, inps, outs, _ = self._inst_list[chosen].get()
        assert len(outs) == 1
        out = outs[0]
        out_type = self._tensor_map[out].dtype
        out_device = self._tensor_map[out].device

        out_shape = self._shape(out)
        dim = random.randint(0, len(out_shape) - 1)
        # if isinstance(op, MatMul):
        loop = SpatialLoop('i', dim, out_shape, out_type, out_device)
        self._inst_list[chosen].set_dissolve(loop)
        return chosen

class ModThenRecover(object):
    pass