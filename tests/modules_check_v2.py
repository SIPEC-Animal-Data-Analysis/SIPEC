import os
import compiler
from compiler.ast import Discard, Const
from compiler.visitor import ASTVisitor

def pyfiles(startPath):
    r = []
    d = os.path.abspath(startPath)
    if os.path.exists(d) and os.path.isdir(d):
        for root, dirs, files in os.walk(d):
            for f in files:
                n, ext = os.path.splitext(f)
                if ext == '.py':
                    r.append([d, f])
    return r

class ImportVisitor(object):
    def __init__(self):
        self.modules = []
        self.recent = []
    def visitImport(self, node):
        self.accept_imports()
        self.recent.extend((x[0], None, x[1] or x[0], node.lineno, 0)
                           for x in node.names)
    def visitFrom(self, node):
        self.accept_imports()
        modname = node.modname
        if modname == '__future__':
            return # Ignore these.
        for name, as_ in node.names:
            if name == '*':
                # We really don't know...
                mod = (modname, None, None, node.lineno, node.level)
            else:
                mod = (modname, name, as_ or name, node.lineno, node.level)
            self.recent.append(mod)
    def default(self, node):
        pragma = None
        if self.recent:
            if isinstance(node, Discard):
                children = node.getChildren()
                if len(children) == 1 and isinstance(children[0], Const):
                    const_node = children[0]
                    pragma = const_node.value
        self.accept_imports(pragma)
    def accept_imports(self, pragma=None):
        self.modules.extend((m, r, l, n, lvl, pragma)
                            for (m, r, l, n, lvl) in self.recent)
        self.recent = []
    def finalize(self):
        self.accept_imports()
        return self.modules

class ImportWalker(ASTVisitor):
    def __init__(self, visitor):
        ASTVisitor.__init__(self)
        self._visitor = visitor
    def default(self, node, *args):
        self._visitor.default(node)
        ASTVisitor.default(self, node, *args) 

def parse_python_source(fn):
    contents = open(fn, 'rU').read()
    ast = compiler.parse(contents)
    vis = ImportVisitor() 
    compiler.walk(ast, vis, ImportWalker(vis))
    return vis.finalize()

for d, f in pyfiles('../SwissKnife'):
    print d, f
    print parse_python_source(os.path.join(d, f)) 
