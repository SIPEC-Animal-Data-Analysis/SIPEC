import os
import ast
import glob
from collections import defaultdict
import importlib
import sys

sys.path.insert(1,os.path.join('./SwissKnife/'))
sys.path.insert(1,os.path.join('./SwissKnife/mrcnn/'))

def main():
    import_dict = []
    from_dict = []
    for file in glob.glob("/home/user/SIPEC/SwissKnife/*.py"):
    #for file in glob.glob("./test.py"):
        with open(file, "r") as source:
            tree = ast.parse(source.read())
        analyzer = Analyzer()
        analyzer.visit(tree)
        data = analyzer.data()
        import_dict.append(data['import'])
        from_dict.append(data['from'])
    dd = defaultdict(list)
    import_list = set([item for sublist in import_dict for item in sublist])
    for d in from_dict:
        for key, value in d.items():
            dd[key].append(value)
    for key, value in dd.items():
        dd[key] = set([item for sublist in value for item in sublist])
    load_modules(import_list, dd)

def load_modules(import_list, dd):
    for mod in import_list:
        print("mod {}".format(mod))
        importlib.import_module(mod)
        print(f"Importing module: {mod}")
    for key, value in dd.items():
        if key == "SwissKnife.mrcnn":
            pass
        else:
            tmp_mod = importlib.import_module(key)
            for i in value:
                print("tmp_mod, i : {} {}".format(tmp_mod, i))
                getattr(tmp_mod, i)
                print(f"Importing: {i} from: {tmp_mod}")

class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": defaultdict(list)}

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["import"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"][node.module].append(alias.name)
        self.generic_visit(node)

    def data(self):
        return self.stats

if __name__=="__main__":
    print(os.getcwd())
    main()

