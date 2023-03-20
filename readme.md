A fuzzing tool built upon NNSmith to find bugs in the new compilers of PyTorch, covering more dynamic features to be tested (control flow, loops, tensor manipulation, etc). Already found 10+ bugs of PyTorch compiler (see https://github.com/pytorch/pytorch/issues/created_by/soodoshll and https://github.com/openai/triton/issues/created_by/soodoshll)


# Dependencies

 * newest PyTorch compiled from source
 * NNSmith

# How to Use

Add this directory to `PATH`. Then run

```
python -m temisu.fuzz
```

# Features

Currently support testing of (dynamic) features:

 - if statement
 - for loop
 - list comprehension
 - nested function
 - inplace tensor mutation

# Transformations

Transformations that guarantee EMI (equivalence modulo inputs)

## True Conditional Blocks

Inserting true conditions

## Operator Resolution

 * Elementwise
 * Reduction
 * MatMul

## Modify then recover

 * Matmul then inverse
 * Backup, modify then recover

## Data Movement
 * Inplace Operator
 * Offloading
 * Gradient Checkpointing

## Functionalize
 * Choose some instructions to form a new (sub)function

## Compound data types test
 * Tuple
 * List
 * Dict