from functools import partial
from typing import Type

import torch
import nnsmith

import ast
from collections.abc import Iterable
from enum import Enum

DissolveType = Enum('DissolveType', ['ELEMENTWISE'])


class Guard(object):
    def __init__(self, expr, first_inst=None):
        self._expr = expr
        self._first_inst = first_inst 
    
    def get_first_inst(self):
        return self._first_inst

    def get(self):
        return self._expr

class SpatialLoop(object):
    def __init__(self, idx, dim, shape, dtype, device='cuda'):
        self._idx = idx
        self._dim = dim
        self._shape = shape
        self._ndim = len(shape)
        self._dtype = dtype
        self._device = device
    
    def get_index_str(self):
        ret = [':'] * self._ndim
        ret[self._dim] = self._idx
        return '[' + ','.join(ret) + ']'
    
    def get_init_expr(self):
        return f"torch.empty({self._shape}, dtype={self._dtype}, device='{self._device}')"
    
    def get_loop(self):
        return f"for {self._idx} in range(0, {self._shape[self._dim]}):"

class MutatorInstruction(object):
    def __init__(self, inst, inps, outs, op, guard=None, inner_guard=None):
        self._inst = inst
        self._inps = inps
        self._outs = outs
        self._op = op
        self._guard = guard
        self._inner_guard = inner_guard

        self._dissolve = None
    
    def get(self):
        return self._inst, self._inps, self._outs, self._op
    
    def set_guard(self, guard):
        self._guard = guard
    
    def get_guard(self):
        return self._guard
    
    def set_dissolve(self, dissolve_type):
        self._dissolve = dissolve_type

    def get_dissolve(self):
        return self._dissolve
    
    def set_inner_guard(self, guard):
        self._inner_guard = guard

    def get_inner_guard(self):
        return self._inner_guard

class Render(object):
    def __init__(self, model, instructions):
        self._defs = []
        self._code = []
        self._model = model
        self._instructions = instructions
        
        self._support_op = [method[len("render"):] for method in dir(self) if method.startswith("render")]
        self._ast = None

    def run(self):
        self._code.clear()
        model = self._model
        stmt_idx = 0
        while stmt_idx < len(self._instructions):
            minst = self._instructions[stmt_idx]
            guard = minst.get_guard()
            if guard is not None and minst is not guard.get_first_inst():
                continue
            if guard is None:
                ret = self._render_single_inst(minst)
                self._code.extend(ast.parse(ret).body) 
                stmt_idx += 1
            else:
                cond = guard.get()
                if_stmt = f"if {cond}:\n\tpass"
                if_stmt = ast.parse(if_stmt).body[0]
                body = []
                
                while stmt_idx < len(self._instructions) and \
                    self._instructions[stmt_idx].get_guard() is guard:
                        inst = self._render_single_inst(self._instructions[stmt_idx])
                        body.extend(ast.parse(inst).body)
                        stmt_idx += 1
                
                if_stmt.body = body
                self._code.append(if_stmt)
        return self._code
    