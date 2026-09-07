"""What matchmaker offers, as the benchmark needs to see it.

Every method-specific fact the benchmark used to hardcode — which methods
exist, which processor each runs with, which class implements it, what its
default kwargs are — now comes from matchmaker's own spec
(``matchmaker/methods.yaml``, interpreted by ``matchmaker.registry``). This
module is the single seam between the two, so that adding a method to
matchmaker makes it appear in the benchmark's ``--method`` choices, its sweeps
and its equivalence check without another edit here.
"""

from typing import List, Optional

from matchmaker.features.audio import FRAME_RATE, SAMPLE_RATE
from matchmaker.matchmaker import AVAILABLE_METHODS, DEFAULT_KWARGS
from matchmaker.registry import REGISTRY

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "AUDIO_FRAME_RATE",
    "audio_rates",
    "available_methods",
    "builtin_methods",
    "default_kwargs",
    "is_builtin",
    "processor_for",
    "resolve_class_values",
    "spec_for",
]

#: Matchmaker's own audio defaults. Referenced rather than repeated so the
#: benchmark's reported sample/frame rate cannot drift from what it ran.
AUDIO_SAMPLE_RATE = SAMPLE_RATE
AUDIO_FRAME_RATE = FRAME_RATE


def available_methods(input_type: str) -> List[str]:
    """Every method ``Matchmaker(input_type=...)`` accepts, registered ones included.

    Read at call time: a submission registers itself on import, so a list
    captured at import time would miss it.
    """
    return list(AVAILABLE_METHODS.get(input_type, []))


def builtin_methods(input_type: str) -> List[str]:
    """The methods declared in matchmaker's spec, without registered ones."""
    return list(REGISTRY.methods.get(input_type, {}))


def is_builtin(input_type: str, method: str) -> bool:
    return method in REGISTRY.methods.get(input_type, {})


def spec_for(input_type: str, method: str):
    """The ``MethodSpec`` for a built-in method (raises for a registered one)."""
    return REGISTRY.method(input_type, method)


def default_kwargs(input_type: str, method: str) -> dict:
    """A private copy of the method's defaults, safe to mutate."""
    return dict(DEFAULT_KWARGS.get(input_type, {}).get(method, {}))


def processor_for(input_type: str, method: str) -> Optional[str]:
    """The feature processor the method runs with by default.

    Falls back to the input type's default, which is what ``Matchmaker`` does
    when a method declares no processor of its own.
    """
    if is_builtin(input_type, method):
        return REGISTRY.default_processor_of(input_type, method)
    declared = DEFAULT_KWARGS.get(input_type, {}).get(method, {}).get("processor")
    return declared or REGISTRY.default_processor.get(input_type)


#: Keys whose value is a class rather than a number. A sweep config can only
#: carry strings, so a bare class name arriving in kwargs is resolved here.
CLASS_VALUED = {
    "tempo_model": "matchmaker.utils.tempo_models",
}


def resolve_class_values(kwargs: dict) -> dict:
    """Turn bare class names in ``kwargs`` into the classes they name.

    A wandb sweep can vary ``tempo_model`` only as a string; matchmaker's
    followers want the class. The spec expresses the same thing with its
    ``!obj`` tag, so this is the sweep-side equivalent rather than a second
    table of models to keep in step.
    """
    import importlib

    out = dict(kwargs)
    for key, module_name in CLASS_VALUED.items():
        value = out.get(key)
        if not isinstance(value, str):
            continue
        module = importlib.import_module(module_name)
        try:
            out[key] = getattr(module, value)
        except AttributeError:
            available = [
                n for n in dir(module) if n.endswith("TempoModel") or "Model" in n
            ]
            raise ValueError(
                f"unknown {key} '{value}'. Available in {module_name}: "
                f"{sorted(available)}"
            ) from None
    return out


def audio_rates(kwargs: dict) -> dict:
    """``{"sample_rate", "frame_rate"}`` for an audio run, as Matchmaker resolves them.

    ``hop_length`` wins over ``frame_rate`` — the same precedence as
    ``Matchmaker.__init__`` — so a config built from these reports the rate the
    run actually used rather than the one nominally requested.
    """
    sample_rate = kwargs.get("sample_rate", AUDIO_SAMPLE_RATE)
    hop_length = kwargs.get("hop_length")
    if hop_length:
        frame_rate = sample_rate / int(hop_length)
    else:
        frame_rate = kwargs.get("frame_rate", AUDIO_FRAME_RATE)
    return {"sample_rate": sample_rate, "frame_rate": frame_rate}
