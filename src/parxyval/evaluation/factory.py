from typing import Callable, Optional

METRIC_REGISTRY = {}


def get_metric(name: str) -> Optional[Callable]:
    name = name.lower().strip().replace("-", "_").replace(" ", "_")
    return METRIC_REGISTRY.get(name, None)


def get_metrics_name():
    return list(METRIC_REGISTRY.keys())


def register_metric(name: str):
    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        return cls

    return decorator
