class Unit:
    def __init__(self, unit_type):
        assert unit_type in ['root', 'node', 'edge', 'start_graph', 'end_graph']
        self.unit_type = unit_type
        self._prev = self
        self._next = self

    @property
    def next(self):
        return self._next
    
    @property
    def prev(self):
        return self._prev
    
    def _remove_from_list(self):
        p, n = self._prev, self._next
        p._next, n._prev = n, p
    
    def prepend(self, x: 'Unit'):
        x._remove_from_list()
        p = self._prev
        p._next, x._prev = x, p
        x._next, self._prev = self, x

    def append(self, x: 'Unit'):
        self._next.prepend(x)

class Node(Unit):
    def __init__(self, unit_type, name, node_type, hierarchy, inputs, **kwargs):
        super(Node, self).__init__(unit_type)
        self.name = name
        self.type = node_type
        self.hierarchy = hierarchy
        self.inputs = inputs
        self.extra_attr_keys = kwargs.keys() if kwargs else []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        ret = f'<node name={self.name} type={self.type} hierarchy={self.hierarchy} inputs={self.inputs}'
        for key in self.extra_attr_keys:
            ret += f' {key}={getattr(self, key)}'
        ret += '/>'
        return ret
    
class Edge(Unit):
    def __init__(self, unit_type, source, target, dtype=None, shape=None):
        super(Edge, self).__init__(unit_type)
        self.source = source
        self.target = target
        self.dtype = dtype
        self.shape = shape

    def __repr__(self) -> str:
        return f'<edge soure={self.source} target={self.target}/>'
