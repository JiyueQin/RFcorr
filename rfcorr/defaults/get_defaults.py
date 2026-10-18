import tomllib
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Mapping


@lru_cache(maxsize=None)
def get_defaults() -> Mapping:
    """
    Load the package-wide settings stored in rfcorr/defaults/defaults.toml.

    The file is read once and cached. Lists are returned as tuples and tables as
    read-only mappings, so the cached settings cannot be changed by a caller.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    None

    ---------------------------------------------------------------------------
    OUTPUT:
    defaults : Mapping
        Read-only mapping with keys:
        'se_methods' : tuple of str
            Standard error methods accepted by rc.corr, named as in the paper.
        'se_method_aliases' : Mapping[str, str]
            Earlier method names mapped to the names in 'se_methods'.
        'rho_zero_dep_models' : tuple of str
            Dependence designs kept only with rho = 0 by rc.make_setting_df.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    text = files("rfcorr.defaults").joinpath("defaults.toml").read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    defaults = {
        key: tuple(value) if isinstance(value, list)
        else MappingProxyType(value) if isinstance(value, dict)
        else value
        for key, value in raw.items()
    }
    return MappingProxyType(defaults)
