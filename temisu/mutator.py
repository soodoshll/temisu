import torch
import random
from functools import cache
from enum import Enum

from nnsmith.abstract.op import *

from .ir import *

ScalarizeMethod = Enum('ScalarizeMethod', ['ELEMENT', 'SHAPE', 'MAX', 'MIN', 'SUM', 'MEAN'])

def _scalarize(k:str, v:torch.Tensor):
    if len(v.shape) == 0:
        return k, v
    p = random.choice(list(ScalarizeMethod))
    if p == ScalarizeMethod.SHAPE:
        dim = random.randint(0, len(v.shape) - 1)
        k = k + f".shape[{dim}]"
        v = v.shape[dim]
    elif p == ScalarizeMethod.ELEMENT:
        dim = Mutator._choose_pos(v)
        k = f"{k}[{','.join(map(str, dim))}]" 
        v = v[dim]
    elif p == ScalarizeMethod.MIN:
        k  = f"{k}.min()"
        v = v.min()
    elif p == ScalarizeMethod.MAX:
        k = f"{k}.max()"
        v = v.max()
    elif p == ScalarizeMethod.SUM:
        k = f"{k}.sum()"
        v = v.sum()
    elif p == ScalarizeMethod.MEAN:
        if not torch.is_floating_point(v):
            k = f"{k}.float()"
            v = v.float()
        k = f"{k}.mean()"
        v = v.mean()

    return k, v

def _generate_true_condition(tensor_map):
    (k1, v1), (k2, v2) = random.sample(tensor_map.items(), 2)

    k1, v1 = _scalarize(k1, v1)
    k2, v2 = _scalarize(k2, v2)
    
    op = '<' if v1 < v2 else '>='
    expr = f"{k1} {op} {k2}"
    return expr

class Mutator(object):
    def __init__(self, tfunc, inputs):
        self._tfunc:TFunction = tfunc
        self._inputs = inputs
        self._mlist = tfunc._model.mlist

        self._stmt_list = tfunc.get_inst_list()
        self._cached_profile = None
    
    def _init_locals(self):
        local_dict = {}
        
        local_dict['mlist'] = self._mlist
        for k,v in self._inputs.items():
            local_dict[k] = v

        return local_dict

    # Be cautious about cache!
    def profile(self):
        if self._cached_profile is not None:
            return self._cached_profile
        local_dict = self._init_locals()
        tensor_table = []
        render = Render(self._tfunc._model, self._stmt_list)
        for stmt in self._stmt_list:
            var = stmt.get()[2][0]
            rendered = render._render_single_inst(stmt)
            exec(rendered,  globals(), local_dict)
            tensor_table.append((var, local_dict[var]))
        self._cached_profile = tensor_table
        return tensor_table

    def insert_tcb(self):
        n_stmts = len(self._stmt_list)
        start = random.randint(0, n_stmts - 1)
        tensor_table = self.profile()[:start]

        available_tensors = {k : v for k, v in tensor_table}
        for k, v in self._inputs.items():
            available_tensors[k] = v

        if len(available_tensors) < 2:
            return self._tfunc

        cond = _generate_true_condition(available_tensors)
        
        end = random.randint(start + 1, n_stmts)
        g = Guard(cond, self._stmt_list[start])
        for inst in self._stmt_list[start:end]:
            inst.set_guard(g)

        return self._tfunc
        
    @staticmethod
    def _choose_pos(t):
        return tuple([random.randint(0, d - 1) for d  in t.shape])
    
    def desolve_op(self):
        profile = self.profile()
        tensor_map = {k : v for k, v in profile}
        for k, v in self._inputs.items():
            tensor_map[k] = v
    
        desolver = OpDesolver(self._tfunc, tensor_map)
        chosen = desolver.run()

        # inner_guard
        if chosen is not None:
            profile = profile[:chosen] 
            tensor_map = {k : v for k, v in profile}
            for k, v in self._inputs.items():
                tensor_map[k] = v
            if len(tensor_map) >= 2:
                cond = _generate_true_condition(tensor_map)
                g = Guard(cond, self._stmt_list[chosen])
                self._stmt_list[chosen].set_inner_guard(g)

        return self._tfunc

    def origin(self):
        return self._tfunc

    def _insert_input_to_profile(self, profile):
        tensor_map = {k : v for k, v in profile}
        for k, v in self._inputs.items():
            tensor_map[k] = v        
        return tensor_map

    def matmul_then_inverse(self):
        profile = self.profile()
        candidates = []
        for i, (_, v) in enumerate(profile):
            if v.dtype in [torch.float32, torch.float64] and len(v.shape) > 0:
                candidates.append(i)
        
        if len(candidates) == 0:
            return self._tfunc
        
        chosen = random.choice(candidates)
        var, val = profile[chosen]
        dim = val.shape[-1]

        inst_list = self._stmt_list 

        a = torch.rand([dim, dim], dtype=val.dtype, device=val.device)
        a_module = ConstFn(a)
        self._mlist.append(a_module)
        init_stmt = MutatorInstruction(a_module, [], ['tmp'], Constant(a.shape))
        inst_list.insert(chosen + 1, init_stmt)
        matmul_stmt = MutatorInstruction(torch.matmul, [var, 'tmp'], [var], MatMul())
        inst_list.insert(chosen + 2, matmul_stmt)

        dep = None
        for i in range(chosen + 3, len(inst_list)):
            if var in inst_list[i]._inps:
                dep = i
                break
        if var in self._tfunc._model.output_map:
            dep = len(inst_list)
        if dep is not None:
            a_inv = torch.inverse(a)
            a_inv_module = ConstFn(a_inv)
            self._mlist.append(a_inv_module)
            inv_init_stmt = MutatorInstruction(a_inv_module, [], ['tmp_inv'], Constant(a.shape))
            inv_stmt = MutatorInstruction(torch.matmul, [var, 'tmp_inv'], [var], MatMul())
            inst_list.insert(dep, inv_init_stmt)
            inst_list.insert(dep + 1, inv_stmt)

        self._cached_profile = None
        return self._tfunc

    transforms = ["origin", 
                #   "matmul_then_inverse",
                #   "insert_tcb", 
                #   "desolve_op",
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