# DFlash Weight Shape Debug

The checkpoint has `fc.weight: [5120, 25600]` but the error shows weight `[25600, 5120]`.

Let me check if the weight is being transposed during SM70 preparation or if the parameter has a different shape.

Actually wait - `ModelWeightParameter` inherits from both `_ColumnvLLMParameter` and `RowvLLMParameter`. Let me check if these have any transpose logic.

Looking at `create_weights`:
```python
weight = ModelWeightParameter(
    data=torch.empty(
        sum(output_partition_sizes),  # 5120
        input_size_per_partition,       # 25600
        dtype=params_dtype,
    ),
    input_dim=1,
    output_dim=0,
    weight_loader=weight_loader,
)
```

So the parameter shape is `[5120, 25600]` which matches the checkpoint.

But the error says `mat1 and mat2 shapes cannot be multiplied (11x5120 and 25600x5120)`.

For `F.linear(x, weight)`:
- x: [11, 5120]
- weight: [25600, 5120]
- F.linear computes `x @ weight.T`
- weight.T: [5120, 25600]
- x @ weight.T: [11, 5120] @ [5120, 25600] = [11, 25600] ✓

This SHOULD work! Unless... the weight in the parameter is actually stored as `[25600, 5120]` (transposed from the checkpoint).

Wait, let me check the SM70 preparation code. In `process_weights_after_loading`:
```python
if SM70_F16_DENSE_ENABLED and is_layer_sm70_f16_dense(prefix):
    prepared = ops.sm70_f16_prepare(layer.weight)
    layer._sm70_f16_tm_weight = prepared[0]
```

The `sm70_f16_prepare` function might transpose the weight! And `_maybe_sm70_dense_forward` might use the transposed weight.

But the error trace shows:
```
File ".../linear.py", line 544, in forward
    output = self.quant_method.apply(self, x, bias)
```

This means `_maybe_sm70_dense_forward` returned None (SM70 path was not taken). So the fallback path is using the original weight.

But then `default_unquantized_gemm` calls `F.linear(x, weight, bias)` with weight `[25600, 5120]`.

HOW is the weight `[25600, 5120]` if the parameter was created as `[5120, 25600]` and loaded from checkpoint `[5120, 25600]`?

The only explanation: somewhere between creation and the forward pass, the weight was transposed.

Let me check if `ModelWeightParameter.__torch_function__` does anything. It just calls `super().__torch_function__`. So no.

Actually, wait. Let me re-read the `BasevLLMParameter.__torch_function__`:
```python
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    return super().__torch_function__(func, types, args, kwargs)
```

This is called when torch operations are performed on the parameter. But it just calls the default behavior.

Hmm, let me check if there's a transpose happening in `dispatch_unquantized_gemm`. No, it just calls `F.linear`.

Wait, I just realized something. The `ModelWeightParameter` is a subclass of both `_ColumnvLLMParameter` and `RowvLLMParameter`. Maybe the MRO (Method Resolution Order) causes some unexpected behavior. Let me check the MRO.

Actually, the simplest thing to do is just print the actual weight shape in the model after loading. Let me run a test script.

Actually, looking more carefully at the error, I notice the weight shape is `[25600, 5120]`. If the parameter was created as `[5120, 25600]` and the weight loader does direct copy, the shape should remain `[5120, 25600]`.

Unless... the `AutoWeightsLoader` is matching `model.fc.weight` against `self.lm_head` somehow? No, that doesn't make sense.

Wait, what if the `fc` weight name in the checkpoint is NOT `fc.weight` but something else? Let me check the checkpoint again.

From the checkpoint listing:
```
fc.weight: torch.Size([5120, 25600])
```

Yes, it's `fc.weight` with shape `[5120, 25600]`.

And the `load_weights` prefixes it to `model.fc.weight`.

And `AutoWeightsLoader(self)` loads into `self.model.fc`.

So the parameter should have shape `[5120, 25600]`.

But the error shows `[25600, 5120]`. There MUST be a transpose somewhere.

Oh wait! I just realized. `RowvLLMParameter` might have a transpose in its `load_row_parallel_weight` method. Let me check.

Actually, looking at `BasevLLMParameter`:
```python
def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
    self._assert_and_load(loaded_weight)
```

This just does direct copy. No transpose.

But wait - `ModelWeightParameter` inherits from both `_ColumnvLLMParameter` and `RowvLLMParameter`. The MRO might pick up methods from one or the other. Let me check if either has a transpose.

Actually, looking at `_ColumnvLLMParameter`:
```python
class _ColumnvLLMParameter(BasevLLMParameter):
    ...
```

It doesn't override `load_row_parallel_weight`. So it uses the base class which does direct copy.

And `RowvLLMParameter`:
```python
class RowvLLMParameter(BasevLLMParameter):
    ...
```

Also doesn't override `load_row_parallel_weight`.

So the weight is loaded directly without transpose.

But then WHY does the error show `[25600, 5120]`?

Let me think about this differently. What if the parameter data itself is transposed? Like, what if `param.data` is a view of some transposed tensor?

Actually, I just need to check what the actual parameter shape is at runtime. Let me write a small script to load the model and inspect the fc weight shape.

Wait, actually, the `ModelWeightParameter` has `input_dim=1` and `output_dim=0`. These attributes might be used by some weight loading logic to determine how to load the weight.

Let me check if the `AutoWeightsLoader` uses these attributes.

Actually, the `AutoWeightsLoader` just iterates over parameters and calls `weight_loader(param, loaded_weight)`. The `weight_loader` is the `ReplicatedLinear.weight_loader` method which does direct copy.

So the shapes must match. If `param.shape = [5120, 25600]` and `loaded_weight.shape = [5120, 25600]`, the copy succeeds. Then `param.shape` should still be `[5120, 25600]`.

Unless... the `process_weights_after_loading` method transposes the weight.

Looking at `UnquantizedLinearMethod.process_weights_after_loading`:
```python
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    ...
    if SM70_F16_DENSE_ENABLED and is_layer_sm70_f16_dense(prefix):
        prepared = ops.sm70_f16_prepare(layer.weight)
        layer._sm70_f16_tm_weight = prepared[0]
        ...
```

The `sm70_f16_prepare` prepares the weight for SM70 fast path. But it stores the prepared weight in `_sm70_f16_tm_weight`, not in `weight`. So `layer.weight` should remain unchanged.

Unless `_maybe_sm70_dense_forward` modifies `layer.weight`? Let me check.

Actually, `_maybe_sm70_dense_forward` is a separate function. Let me check if it uses the prepared weight or the original weight.

OK, I think I need to just check the actual shape at runtime. Let me write a script.

Actually wait - maybe I should look at the problem from a different angle. The error shows:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (11x5120 and 25600x5120)
```

This is from `F.linear(x, weight, bias)`. The error happens inside `torch.nn.functional.linear` which calls `torch.addmm` or similar.

In PyTorch, `F.linear(x, weight)` computes `x @ weight.T` (or `matmul(x, weight.transpose(-2, -1))`).

For this to work with x=[11, 5120] and weight=[25600, 5120]:
- weight.T = [5120, 25600]
- x @ weight.T = [11, 5120] @ [5120, 25600] → works!

But the error says it doesn't work. This means PyTorch is NOT transposing the weight. This only happens if the weight is NOT 2D (or if some special case is triggered).

Wait, unless the `ModelWeightParameter` has a custom shape property that returns a different shape?

Or maybe the parameter is a subclass of `Parameter` and PyTorch's `F.linear` has special handling for `Parameter` subclasses that might not work correctly?

Actually, looking at the error trace again:
```
File "/home/rah/GooseLLM/vllm/model_executor/parameter.py", line 126, in __torch_function__
    return super().__torch_function__(func, types, args, kwargs)
RuntimeError: mat1 and mat2 shapes cannot be multiplied (11x5120 and 25600x5120)
```

The `__torch_function__` is being called for the `F.linear` operation! And it fails inside the torch function dispatch.

This means the `ModelWeightParameter` is somehow causing the `F.linear` to use the wrong shape. But `__torch_function__` just calls the default behavior...

Wait, maybe `super().__torch_function__` is resolving to `_ColumnvLLMParameter.__torch_function__` or `RowvLLMParameter.__torch_function__` instead of the base `Parameter.__torch_function__`. Let me check if these classes have their own `__torch_function__`.

Actually, I don't see any `__torch_function__` in `_ColumnvLLMParameter` or `RowvLLMParameter`. So `super().__torch_function__` would call `BasevLLMParameter.__torch_function__` which just calls `super().__torch_function__` which is `Parameter.__torch_function__`.

Hmm, this is getting complicated. Let me just run a test script to inspect the actual weight shape.
