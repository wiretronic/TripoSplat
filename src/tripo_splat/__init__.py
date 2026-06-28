from .triposplat import TripoSplatPipeline
from .weights import get_weights

def load(device="cuda"):
    return TripoSplatPipeline(device=device, **get_weights())