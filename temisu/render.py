from functools import partial
from typing import Type

import torch
import nnsmith
from nnsmith.abstract.dtype import DTYPE_GEN_INTS
from nnsmith.abstract.op import *
from nnsmith.materialize import framework_operator_impl
from nnsmith.materialize.torch.dialect import Flatten, Linear, TorchReduceSum

from .logging import RENDER_LOG
import ast

# TORCH_REALIZABLE_OPS = FULL_OPERATOR_SETS["core"] + FULL_OPERATOR_SETS["torch"]

class Render(object):
    def __init__(self, model):
        self._defs = []
        self._code = []
        self._model = model
        
        self._support_op = [method[len("render"):] for method in dir(self) if method.startswith("render")]
        
        self._ast = None

    def _get_render_fn(self, op):
        for support_op in self._support_op:
            op_type = globals()[support_op]
            if isinstance(op, op_type):
                return getattr(self, "render"+support_op)
        return None

    def run(self):
        self._code.clear()
        model = self._model
        RENDER_LOG.debug(f"start rendering, inputs:{model.ir.input_var()}")
        for stmt_idx, (inst, inps, outs, op) in enumerate(model.instructions):
            render_fn = self._get_render_fn(op)
            if render_fn is None:
                RENDER_LOG.warning(f"Unsupported Op {op}")
            else:
                ret = render_fn(inst, inps, outs, op)
                if ret is not None:
                    self._code.append(ast.parse(ret).body[0]) 
        return self._stmt_to_func()
    
    def get_last_ast(self):
        return self._ast
    
    def get_last_code(self):
        return ast.unparse(self._ast)

    def _mlist_comments(self):
        ret = ""
        for i, l in enumerate(self._model.mlist):
            ret += f"mlist[{i}] = {l}\n"
        return ret

    def get_full_code(self, mlist_file="mlist.th"):
        comments = '""" Module information:\n' + self._mlist_comments() + '"""\n'
        headers = f"import torch\nmlist=torch.load('{mlist_file}')\n"
        return comments + headers + self.get_last_code()

    def _stmt_to_func(self):
        model = self._model
        _func_def = ast.parse(f"def foobar(mlist, {','.join(model.ir.input_var())}):\n\tpass")

        _ret = 'return ' + ','.join(model.output_map)
        _func_def.body[0].body = self._code + [ast.parse(_ret).body[0]]

        RENDER_LOG.debug(f'rendered model\n{ast.unparse(_func_def)}')
        self._ast = _func_def

        local_dict = {}
        exec(compile(_func_def, "<string>", "exec"), globals(), local_dict)
        func = local_dict['foobar']
        return func

    def _align_inputs(self, *args, **kwargs):
        inputs = {}
        if len(args) == len(self._model.input_map):
            for i, key in enumerate(self._model.ir.input_var()):
                inputs[key] = args[i]
        elif len(kwargs) == len(self._model.input_map):
            for ir_key in self._model.input_map:
                inputs[ir_key] = kwargs[ir_key]
        else:
            raise ValueError("Either user args only or kwargs only")
        return inputs

    @staticmethod
    def _one_one_op(inps, outs, torch_fn):
        assert len(outs) == 1 and len(inps) == 1
        return f"{outs[0]}={torch_fn}({','.join(inps)})"

    @staticmethod
    def _many_one_op(inps, outs, torch_fn):
        assert len(outs) == 1
        return f"{outs[0]}={torch_fn}({','.join(inps)})"

    def _index_module(self, torch_module):
        for i, l in enumerate(self._model.mlist):
            if torch_module is l:
                return i

    def _index_module_and_call(self, inst, inps, outs, op):
        assert len(inps) == 1
        module_id = self._index_module(inst)
        return f"{outs[0]}=mlist[{module_id}]({','.join(inps)})"

    def renderConstant(self, inst, inps, outs, op):
        module = self._index_module(inst)
        return f"{outs[0]} = mlist[{module}]()"
    
    def renderReLU(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.relu")
    
    def renderAdd(self, inst, inps, outs, op):
        assert len(outs) == 1
        return f"{outs[0]}=torch.add({','.join(inps)})"

    def renderGELU(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.nn.functional.gelu")

    def renderLeakyReLU(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.nn.functional.leaky_relu")

    def renderPReLU(self, inst, inps, outs, op):
        assert len(inps) == 1
        module_id = self._index_module(inst)
        return f"{outs[0]}=mlist[{module_id}]({','.join(inps)})"

    def renderSigmoid(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.sigmoid")

    def renderSin(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.sin")

    def renderCos(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.cos")

    def renderAsin(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.asin")

    def renderAcos(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.acos")

    def renderTan(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.tan")

    def renderAtan(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.atan")

    def renderAbs(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.abs")

    def renderWhere(self, inst, inps, outs, op):
        assert len(inps) == 3
        return self._many_one_op(inps, outs, "torch.where")

    def renderSub(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.sub")

    def renderMul(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.mul")

    def renderDiv(self, inst, inps, outs, op):
        if op.input_like[0].dtype in DTYPE_GEN_INTS:
            assert len(inps) == 2
            return f"{outs[0]}=torch.div({inps[0]}, {inps[1]}, rounding_mode='floor')" 
        return self._many_one_op(inps, outs, "torch.div")

    def renderMax(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.max")

    def renderMin(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.min")

    def renderEqual(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.eq")

    def renderGreater(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.gt")

    def renderLess(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.lt")

    def renderAnd(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.logical_and")

    def renderOr(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.logical_or")

    def renderXor(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.logical_xor")

    def renderPow(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.logical_pow")

    def renderFloor(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.floor")

    def renderCeil(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.ceil")

    def renderClip(self, inst, inps, outs, op):
        if op.input_like[0].dtype in DTYPE_GEN_FLOATS:
            return f"{outs[0]}=torch.clip({inps[0]}, -1.5, 1.5)"
        else:
            return f"{outs[0]}=torch.clip({inps[0]}, -1, 1)"

    def renderRound(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.round")

    def renderSqrt(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.sqrt")

    def renderLog2(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.log2")

    def renderNeg(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.neg")

    def renderSoftmax(self, inst, inps, outs, op):
        return f"{outs[0]}=torch.nn.functional.softmax({inps[0]}, dim={op.dim})"

    def renderNeg(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, "torch.neg")

    def renderMaxPool2d(self, inst, inps, outs, op):
        return f"""{outs[0]} = torch.nn.functional.max_pool2d({inps[0]},
            kernel_size=({op.kernel_h_size}, {op.kernel_w_size}),
            stride={op.stride},
            padding={op.padding})"""
    
    def renderAvgPool2d(self, inst, inps, outs, op):
        return f"""{outs[0]} = torch.nn.functional.avg_pool2d({inps[0]},
            kernel_size=({op.kernel_h_size}, {op.kernel_w_size}),
            stride={op.stride},
            padding={op.padding})"""

    def renderSlice(self, inst, inps, outs, op):
        reg = op.extra_attrs["region"]
        shape = op.input_like[0].shape
        dim_s = shape[op.extra_attrs["axis"]]
        start, end = op.start, op.end
        if reg in ["left", "mid"]:
            start -= dim_s
        # actual end would be 0, which is not really 'left'
        if reg == "left" and end < dim_s and end != Slice.INT_MAX:
            end -= dim_s
        idx = tuple(
            ":" if i != op.extra_attrs["axis"] else f"{start}:{end}:{op.step}"
            for i in range(op.extra_attrs["ndims"])
        )
        return f"{outs[0]} = {inps[0]}[{','.join(idx)}]"

    def renderPad(self, inst, inps, outs, op):
        padding_str = ','.join(map(str, op.padding_list))
        if op.extra_attrs["type"] == "constant":
            stmt = f"{outs[0]} = torch.nn.functional.pad({inps[0]}, ({padding_str},), 'constant', value=0.5)"
        elif op.extra_attrs["type"] == "replicate" or op.extra_attrs["type"] == "reflect":
            stmt = f"{outs[0]} = torch.nn.functional.pad({inps[0]}, ({padding_str},), '{op.extra_attrs['type']}')"
        return stmt

    def renderExpand(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.output_like[0].shape))
        return f"{outs[0]} = {inps[0]}.expand({shape_str})"

    def renderBatchNorm2d(self, inst, inps, outs, op):
        module_idx = self._index_module(inst)
        return f"{outs[0]} = mlist[{module_idx}]({inps[0]})"
    
    def renderConv1d(self, inst, inps, outs, op):
        return self._index_module_and_call(inst, inps, outs, op)

    def renderNCHWConv2d(self, inst, inps, outs, op):
        return self._index_module_and_call(inst, inps, outs, op)

    def renderReshape(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.target_shape))
        return f"{outs[0]} = {inps[0]}.reshape({shape_str})"

    def renderFlatten(self, inst, inps, outs, op):
        return self._one_one_op(inps, outs, "torch.Tensor.flatten")

    def renderTranspose(self, inst, inps, outs, op):
        dim0, dim1 = op._init_swap_dims(op.input_like[0].shape)
        return f"{outs[0]} = {inps[0]}.transpose({dim0},{dim1})"

    def renderNearestInterp(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.size))
        return f"{outs[0]} = torch.nn.functional.interpolate({inps[0]}, size=({shape_str}), mode='nearest')"

    def renderLinearInterp(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.size))
        return f"{outs[0]} = torch.nn.functional.interpolate({inps[0]}, size=({shape_str}), mode='linear')"

    def renderBilinearInterp(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.size))
        return f"{outs[0]} = torch.nn.functional.interpolate({inps[0]}, size=({shape_str}), mode='bilinear')"

    def renderBicubicInterp(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.size))
        return f"{outs[0]} = torch.nn.functional.interpolate({inps[0]}, size=({shape_str}), mode='bicubic')"

    def renderTrilinearInterp(self, inst, inps, outs, op):
        shape_str = ','.join(map(str, op.size,))
        return f"{outs[0]} = torch.nn.functional.interpolate({inps[0]}, size=({shape_str}), mode='trilinear')"
    
    def renderSqueeze(self, inst, inps, outs, op):
        return f"{outs[0]} = {inps[0]}.squeeze({op.extra_attrs['reduce_dim']})"
    
    def _reduceOp(self, inps, outs, op, torch_fn):
        dims = op.extra_attrs["reduce_dim"]
        if isinstance(dims, int):
            dims = [dims]
        shape_str = ','.join(map(str, dims)) 
        return f"{outs[0]} = {inps[0]}.{torch_fn}({shape_str})" 
    
    def renderTorchReduceSum(self, inst, inps, outs, op):
        return self._reduceOp(inps, outs, op, 'sum')

    def renderReduceMin(self, inst, inps, outs, op):
        stmt = self._reduceOp(inps, outs, op, 'min')
        if op.extra_attrs["reduce_dim"] is not None:
            stmt += '.values'
        return stmt

    def renderReduceMax(self, inst, inps, outs, op):
        stmt = self._reduceOp(inps, outs, op, 'max')
        if op.extra_attrs["reduce_dim"] is not None:
            stmt += '.values'
        return stmt
    
    def renderReduceMean(self, inst, inps, outs, op):
        return self._reduceOp(inps, outs, op, 'mean')

    def renderArgMin(self, inst, inps, outs, op):
        return self._reduceOp(inps, outs, op, 'argmin')

    def renderArgMax(self, inst, inps, outs, op):
        return self._reduceOp(inps, outs, op, 'argmax')

    def renderTril(self, inst, inps, outs, op):
        return f"{outs[0]} = {inps[0]}.tril({op.diagonal})"

    def renderTriu(self, inst, inps, outs, op):
        return f"{outs[0]} = {inps[0]}.triu({op.diagonal})"

    def renderLinear(self, inst, inps, outs, op):
        return self._index_module_and_call(inst, inps, outs, op)
    
    def renderConcat(self, inst, inps, outs, op):
        axis = op.extra_attrs["axis"]
        return f"{outs[0]} = torch.cat(({','.join(inps)},), dim={axis})"

    def renderMatMul(self, inst, inps, outs, op):
        return self._many_one_op(inps, outs, 'torch.matmul')

    def renderCast(self, inst, inps, outs, op):
        return f"{outs[0]} = {inps[0]}.to(dtype={op.extra_attrs['to'].torch()})"

