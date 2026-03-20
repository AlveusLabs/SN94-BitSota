__spec_version__ = 42


def __getattr__(name: str):
    if name == "BittensorNetwork":
        from ._singleton import BittensorNetwork

        return BittensorNetwork
    raise AttributeError(name)
