"""
rfcorr: correlation between two random fields (maps), with standard errors that
account for spatial dependence.

Every function is available as ``rc.<function>`` after ``import rfcorr as rc``,
wherever it lives in the package.
"""
from .defaults import *
from .fields import *
from .estimation import *
from .simulation import *
