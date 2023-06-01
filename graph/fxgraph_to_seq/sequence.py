import builtins
import inspect
import sys
from collections import deque
from typing import Any, Callable, Dict

import torch
import torch.fx
from torch.nn.modules.container import ModuleDict, ModuleList, Sequential

from .unit import Edge, Node, Unit


def _find_module_of_method(orig_method: Callable[..., Any]) -> str:
    name = orig_method.__name__
    module = orig_method.__module__
    if module is not None:
        return module
    for guess in [torch, torch.nn.functional]:
        if getattr(guess, name, None) is orig_method:
            return guess.__name__
    raise RuntimeError(f'cannot find module for {orig_method}')

def _get_qualified_name(func: Callable[..., Any]) -> str:
    if getattr(builtins, func.__name__, None) is func:
        return func.__name__
    name = func.__name__
    module = _find_module_of_method(func)
    module = module.replace('torch._ops', 'torch.ops')
    return f'{module}.{name}'

def _pretty_print_type(node_type):
    if isinstance(node_type, str):
        return node_type
    if hasattr(node_type, '__module__'):
        if hasattr(node_type, '__name__'):
            if node_type.__module__ == 'builtins':
                return f'builtins.{node_type.__name__}'
            elif node_type.__module__ == '_operator':
                return f'operator.{node_type.__name__}'
        else:
            return f'{node_type.__module__}.{node_type.__class__.__name__}'
    if isinstance(node_type, Callable):
        try:
            # there maybe RuntimeError when executing `_find_module_of_method`
            return _get_qualified_name(node_type)
        except RuntimeError:
            # workaround for built-in method apply of FunctionMeta
            print(f'INFO: _get_qualified_name failed when trying {node_type}')
            return type(node_type).__name__
    else:
        return type(node_type).__name__

def _format_target(base: str, target: str) -> str:
    elems = target.split('.')
    r = base
    for e in elems:
        if not e.isidentifier():
            r = f'getattr({r}, "{e}")'
        else:
            r = f'{r}.{e}'
    return r

def _format_module_type(module) -> str:
    return module.__class__.__module__ + '.' + module.__class__.__name__

def is_leaf_module(m: torch.nn.Module, leaf_module=()) -> bool:
    if isinstance(m, leaf_module):
        return True
    return (m.__module__.startswith('torch.nn') and not isinstance(m, (Sequential, ModuleList, ModuleDict)))

class StartGraph(Unit):
    def __init__(self, name, graph_type):
        super(StartGraph, self).__init__('start_graph')
        self.name = name
        self.type = graph_type

    def __repr__(self) -> str:
        return f'<graph name={self.name} type={self.type}'

class EndGraph(Unit):
    def __init__(self, start):
        super(EndGraph, self).__init__('end_graph')
        self.start = start # point at the correpsonding StartGraph unit

    def __repr__(self) -> str:
        return '</graph>'

class _unit_list:
    def __init__(self, graph: 'Sequence', direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            yield cur
            cur = getattr(cur, direction)

class _node_list:
    def __init__(self, graph: 'Sequence', direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            while cur.unit_type != 'node' and cur is not root:
                cur = getattr(cur, direction)
            if cur is root:
                break
            yield cur
            cur = getattr(cur, direction)

class _edge_list:
    def __init__(self, graph: 'Sequence', direction: str = '_next'):
        assert direction in ['_next', '_prev']
        self.graph = graph
        self.direction = direction

    def __iter__(self):
        root, direction = self.graph._root, self.direction
        cur = getattr(root, direction)
        while cur is not root:
            while cur.unit_type != 'edge' and cur is not root:
                cur = getattr(cur, direction)
            if cur is root:
                break
            yield cur
            cur = getattr(cur, direction)

class Sequence:
    def __init__(self, gm: torch.fx.GraphModule, orig_module: torch.nn.Module, with_config=False):
        self.gm = gm
        self.orig_module = orig_module
        self.with_config = with_config
        self.name = self.gm.__class__.__name__
        self._root : Unit = Unit('root')
        self._insert = self._root.prepend
        self.name_to_startgraph : Dict[str, StartGraph] = {}
        self.compile_from_gm()

    @property
    def units(self):
        return _unit_list(self)

    @property
    def nodes(self):
        return _node_list(self)

    @property
    def edges(self):
        return _edge_list(self)
    
    def create_node(self, name, node_type, hierarchy, inputs, **kwargs):
        n = Node('node', name, node_type, hierarchy, inputs, **kwargs)
        self._insert(n)
        return n
    
    def create_edge(self, source, target, dtype=None, shape=None):
        e = Edge('edge', source, target, dtype, shape)
        self._insert(e)
        return e
    
    def start_graph(self, name: str, graph_type):
        # TODO: validity check
        g = StartGraph(name, graph_type)
        self._insert(g)
        self.name_to_startgraph[name] = g
        return g
    
    def end_graph(self, start: StartGraph):
        # TODO: validity check
        g = EndGraph(start)
        self._insert(g)
        return g
    
    def compile_from_gm(self):
        # to handle hierarchy
        def common_prefix(a, b):
            if a == b:
                return a
            cp = ''
            a_atoms = a.split('.')
            b_atoms = b.split('.')
            for a_atom, b_atom in zip(a_atoms, b_atoms):
                if a_atom == b_atom:
                    if len(cp) == 0:
                        cp += a_atom
                    else:
                        cp += '.' + a_atom
                else:
                    return cp
            return cp

        def graph_relationship(a, b):
            r'''Returns the relationship between graph hierarchy a and b

            Example::

                >>> a = 'A.b'
                >>> b = 'A.b.c'
                >>> graph_relationship(a, b)
                child
            '''
            a_atoms = a.split('.')
            b_atoms = b.split('.')
            cp = common_prefix(a, b)
            relationship = ''
            if len(b_atoms) > len(a_atoms):
                if len(b_atoms) - len(a_atoms) == 1:
                    if cp == a:
                        relationship = 'child'
                    else:
                        relationship = 'parallel'
                else:
                    if cp == a:
                        relationship = 'descendant'
                    else:
                        relationship = 'parallel'
            elif len(b_atoms) < len(a_atoms):
                if len(a_atoms) - len(b_atoms) == 1:
                    if cp == b:
                        relationship = 'parent'
                    else:
                        relationship = 'parallel'
                else:
                    if cp == b:
                        relationship = 'ancestor'
                    else:
                        relationship = 'parallel'
            else:
                if a == b:
                    relationship = 'same'
                else:
                    relationship = 'parallel'
            return relationship, cp

        fxnode2name = {}
        hierarchy_stack = deque()
        
        for node in self.gm.graph.nodes:
            # traverse the fx graph nodes to generate sequence nodes
            # handle node type
            if isinstance(node.target, str):
                if node.op == 'placeholder':
                    # node_type = node.target + '(placeholder)'
                    node_type = 'placeholder'
                elif node.op == 'output':
                    node_type = node.target
                else:
                    # [get_attr, call_module, call_method]
                    try:
                        target_obj = eval(_format_target('self.gm', node.target))
                        node_type = _pretty_print_type(target_obj)
                    except AttributeError:
                        node_type = _pretty_print_type(node.target)
            else:
                # [call_function]
                node_type = _pretty_print_type(node.target)

            # make hierarchy to be the module(with path) where the node operation lies 
            # hierarchy = node_to_originating_module.get(node)
            if node.meta['nn_module_stack']:
                hierarchy = node.meta['nn_module_stack'].popitem()[0]
            else:
                hierarchy = ''
            if node.op == 'call_module':
                hierarchy = '.'.join(hierarchy.split('.')[:-1])
            if hierarchy:
                hierarchy = self.gm.__class__.__name__ + '.' + hierarchy
            else:
                hierarchy = self.gm.__class__.__name__
            
            atoms = hierarchy.split('.')
            if len(atoms) == 1:
                # it's the top graph
                # TODO: improve the naming mechanism
                cur = self.gm.__class__.__name__
                if len(hierarchy_stack) > 0 and cur != hierarchy_stack[-1]:
                    while hierarchy_stack[-1] != cur:
                        name = hierarchy_stack.pop()
                        self.end_graph(self.name_to_startgraph[name])
                elif len(hierarchy_stack) > 0 and cur == hierarchy_stack[-1]:
                    pass
                else:
                    hierarchy_stack.append(cur)
                    graph_type = _format_module_type(self.orig_module)
                    self.start_graph(cur, graph_type)
            else:
                cur = hierarchy
                prefix = self.gm.__class__.__name__
                if len(hierarchy_stack) > 0 and cur == hierarchy_stack[-1]:
                    pass
                elif len(hierarchy_stack) > 0 and cur != hierarchy_stack[-1]:
                    rls, cp = graph_relationship(hierarchy_stack[-1], cur)
                    if rls == 'child':
                        hierarchy_stack.append(cur)
                        obj = eval(_format_target('self.orig_module', cur.replace(prefix + '.', '')))
                        graph_type = _format_module_type(obj)
                        self.start_graph(cur, graph_type)
                    elif rls == 'descendant':
                        temp = cur.replace(hierarchy_stack[-1] + '.', '')
                        next_hierarchy = hierarchy_stack[-1]
                        for atom in temp.split('.'):
                            next_hierarchy += '.' + atom
                            hierarchy_stack.append(next_hierarchy)
                            obj = eval(_format_target('self.orig_module', next_hierarchy.replace(prefix + '.', '')))
                            graph_type = _format_module_type(obj)
                            self.start_graph(next_hierarchy, graph_type)
                    elif rls == 'parent':
                        name = hierarchy_stack.pop()
                        self.end_graph(self.name_to_startgraph[name])
                    elif rls == 'ancestor':
                        temp = hierarchy_stack[-1].replace(cur + '.', '')
                        for _ in temp.split('.'):
                            name = hierarchy_stack.pop()
                            self.end_graph(self.name_to_startgraph[name])
                    elif rls == 'parallel':
                        # keep popping until we meet the common prefix(cp)
                        name = hierarchy_stack[-1]
                        while name != cp:
                            name = hierarchy_stack.pop()
                            self.end_graph(self.name_to_startgraph[name])
                            name = hierarchy_stack[-1]
                        # now we can push the new hierarchy
                        # ! be careful to push it layer by layer
                        temp = cur.replace(hierarchy_stack[-1] + '.', '')
                        next_hierarchy = hierarchy_stack[-1]
                        for atom in temp.split('.'):
                            next_hierarchy += '.' + atom
                            hierarchy_stack.append(next_hierarchy)
                            obj = eval(_format_target('self.orig_module', next_hierarchy.replace(prefix + '.', '')))
                            graph_type = _format_module_type(obj)
                            self.start_graph(next_hierarchy, graph_type)
                else:
                    raise RuntimeError('hierarchy stack is exceptionally empty')
            
            # handle node name
            if node.op == 'placeholder':
                name = node.name
            elif node.op in ['call_module', 'get_attr', 'call_method']:
                name = self.gm.__class__.__name__ + "." + node.target
            elif node.op == 'call_function':
                name = hierarchy + '.' + node.target.__name__
            else:
                name = node.name
            fxnode2name[node] = name

            inputs = []
            if node.args:
                for arg in node.args:
                    try:
                        input_name = fxnode2name.get(arg, arg)
                    except TypeError:
                        input_name = arg
                    inputs.append(input_name)
            inputs = tuple(inputs)
            if self.with_config and node.op == 'call_module':
                module = eval(_format_target('self.orig_module', node.target))
                params = inspect.signature(module.__init__).parameters
                kwargs = {}
                for key in params:
                    value = getattr(module, key, None)
                    if key == 'bias':
                        value = True if value is not None else False
                    kwargs[key] = value
                self.create_node(name, node_type, hierarchy, inputs, **kwargs)
            else:
                self.create_node(name, node_type, hierarchy, inputs)

            # create an edge for each input
            if node.op != "placeholder" and node.args:
                for arg in node.args:
                    try:
                        source = fxnode2name.get(arg, arg)
                        # arg is a Node type
                        try:
                            if arg.meta['type'] is torch.Tensor:
                                dtype = arg.meta['tensor_meta'].dtype
                                shape = arg.meta['tensor_meta'].shape
                            else:
                                dtype = None
                                shape = None
                        except:
                            dtype = None
                            shape = None
                    except TypeError:
                        source = arg
                    self.create_edge(source, name, dtype, shape)

            if node.op == 'output':
                name = hierarchy_stack.pop()
                self.end_graph(self.name_to_startgraph[name])

    def print_tabular(self, file=sys.stdout):
        try:
            from tabulate import tabulate
        except ImportError:
            print("`print_tabular` relies on the library `tabulate`", file=file)
        node_specs = [[n.name, n.hierarchy, _pretty_print_type(n.type), n.inputs]
                      for n in self.nodes]
        print(file=file)
        print('nodes:', file=file)
        print(tabulate(node_specs, headers=['name', 'hierarchy', 'node type', 'inputs']), file=file)
        edge_specs = [[e.source, e.target] for e in self.edges]
        print(file=file)
        print('edges:', file=file)
        print(tabulate(edge_specs, headers=['source', 'target']), file=file)

    def __repr__(self) -> str:
        ret = ''
        for unit in self.units:
            if unit.unit_type == 'node':
                ret += f'<node name="{unit.name}" type="{unit.type}"'
                if unit.extra_attr_keys:
                    for k in unit.extra_attr_keys:
                        ret += f' {k}="{getattr(unit, k)}"'
                ret += '/>'
            elif unit.unit_type == 'edge':
                if self.with_config and unit.dtype:
                    ret += f'<edge source="{unit.source}" target="{unit.target}" dtype="{unit.dtype}" shape="{unit.shape}"/>'
                else:
                    ret += f'<edge source="{unit.source}" target="{unit.target}"/>'
            elif unit.unit_type == 'start_graph':
                ret += f'<graph name="{unit.name}" type="{unit.type}">'
            elif unit.unit_type == 'end_graph':
                ret += '</graph>'
        return ret

def fold_seq(seq: Sequence):
    ret = ''
    unit_iter = iter(seq.units)
    top_graph = next(unit_iter)
    assert top_graph.unit_type == 'start_graph'
    ret += f'<graph name="{top_graph.name}" type="{top_graph.type}">'
    for unit in unit_iter:
        if unit.unit_type == 'node':
            ret += f'<node name="{unit.name}" type="{unit.type}"'
            if unit.extra_attr_keys:
                for k in unit.extra_attr_keys:
                    ret += f' {k}="{getattr(unit, k)}"'
            ret += '/>'
        elif unit.unit_type == 'edge':
            if seq.with_config and unit.dtype:
                ret += f'<edge source="{unit.source}" target="{unit.target}" dtype="{unit.dtype}" shape="{unit.shape}"/>'
            else:
                ret += f'<edge source="{unit.source}" target="{unit.target}"/>'
        elif unit.unit_type == 'start_graph':
            ret += f'<graph name="{unit.name}" type="{unit.type}">'
            next_unit = next(unit_iter)
            while True:
                if next_unit.unit_type != 'end_graph' or next_unit.start != unit:
                    next_unit = next(unit_iter)
                else:
                    ret += '</graph>'
                    break
        elif unit.unit_type == 'end_graph':
            ret += '</graph>'
    return ret
