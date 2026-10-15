"""Configuration: validated dataclasses (:mod:`schema`) and the YAML loader (:mod:`loader`)."""

from vivlmoe.config.loader import (
    config_fingerprint,
    config_from_dict,
    deep_merge,
    from_dict,
    load_config,
    parse_overrides,
    save_resolved,
    to_dict,
)
from vivlmoe.config.registry import Registry
from vivlmoe.config.schema import ConfigError, RunConfig

__all__ = [
    "ConfigError",
    "Registry",
    "RunConfig",
    "config_fingerprint",
    "config_from_dict",
    "deep_merge",
    "from_dict",
    "load_config",
    "parse_overrides",
    "save_resolved",
    "to_dict",
]
