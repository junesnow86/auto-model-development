The most important codes are `concrete_trace_utils` and `fxgraph_to_seq`. `concrete_trace_utils` is used to produce fx graphs, and `fxgraph_to_seq` is used to convert fx graphs into xml form.

Most of trace python scripts in this directory are outdated. They need to be modified to use the latest tracer on [nni](https://github.com/microsoft/nni/tree/master/nni/common/concrete_trace_utils). But notice that there is no `passes` and `kwargs_interpreter.py` on nni. They are used to do shape propagation and counting params & flops. You can refer to `concrete_trace_nlp.py`(which is not outdated) to see the latest usage.

Remeber to set `model.eval()` if you want to check the trace correctness by comparing the output of the original model and the output of the traced GraphModule.