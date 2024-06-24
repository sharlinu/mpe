from importlib.machinery import SourceFileLoader
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    mod = SourceFileLoader('Scenario', pathname ).load_module()
    return mod 
