import rfcorr as rc


def canonical_se_method(se_method: str) -> str:
    """
    Return the standard error method name used in the paper, mapping earlier names to it.

    Development function: a building block of rc.corr_dev; its interface may change.

    The accepted names and the earlier aliases are read from
    rfcorr/defaults/defaults.toml (see rc.get_defaults). Matching is case insensitive.

    ---------------------------------------------------------------------------
    ARGUMENTS:
    se_method : str
        Standard error method: 'general', 'gauss', 'indep', or 'iid', or one of the
        earlier names 'hac', 'hac_gaussian', and 'huristic'.

    ---------------------------------------------------------------------------
    OUTPUT:
    name : str
        The method name in lower case, one of 'general', 'gauss', 'indep', 'iid'.
    ---------------------------------------------------------------------------
    AUTHORS: Jiyue Qin
    ---------------------------------------------------------------------------
    """
    defaults = rc.get_defaults()
    se_methods = defaults["se_methods"]
    name = str(se_method).lower()
    name = defaults["se_method_aliases"].get(name, name)
    if name not in se_methods:
        raise ValueError(f"Unknown se_method {se_method!r}; choose from {se_methods}.")
    return name
