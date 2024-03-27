class NonInstantiableMeta(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError(f"Cannot create instances of {cls.__name__}")
