"""pycel plugin: the Excel functions this workbook family uses that pycel does not ship.

    ExcelCompiler(path, plugins=["presage.ingest.xlfuncs"])

pycel compiles a formula to Python and looks each function name up in its own libs first,
then in the plugin modules. A name it cannot find raises `UnknownFunction` the moment the
cell is recalculated, which in practice means a what-if leaves that cell (and everything
under it) sitting on the value Excel last cached. So a missing function is not a cosmetic
gap: it is a silently stale number. Every function used by `Plan_Model.xlsx` that pycel
lacks is implemented here, and `scratch.ModelGraph` reports anything still missing as
`partial`.

Written in pycel's own house style (`@excel_helper`, `_numerics` for range flattening,
Excel error-code strings rather than exceptions), so these behave like the built-ins:

  QUARTILE / QUARTILE.INC   linear-interpolated percentile, quart 0..4   (Revenue Multiple)
  MEDIAN                    same range, the 50th percentile              (Revenue Multiple)
  RANK / RANK.EQ / RANK.AVG rank within a range, ties share a rank       (the fee models)

QUARTILE is the one the design called out; MEDIAN and RANK turned out to be missing too,
and RANK sits directly downstream of the subscriber-fee lever, so leaving them out would have
made every what-if on that lever partial.
"""

from __future__ import annotations

import numpy as np
from pycel.excellib import _numerics
from pycel.excelutil import (
    EMPTY,
    ERROR_CODES,
    NA_ERROR,
    NUM_ERROR,
    VALUE_ERROR,
    coerce_to_number,
    is_number,
)
from pycel.lib.function_helpers import excel_helper

__all__ = ["quartile", "quartile_inc", "median", "rank", "rank_eq", "rank_avg"]


def _scalar(value, default=None):
    """Excel-style coercion of one argument to a float, or an error-code string."""
    if value is None or value == EMPTY or value == "":
        if default is None:
            return VALUE_ERROR
        return float(default)
    if isinstance(value, str) and value in ERROR_CODES:
        return value
    if isinstance(value, bool):
        return float(value)
    num = coerce_to_number(value)
    if isinstance(num, str) or not is_number(num):
        return VALUE_ERROR
    return float(num)


def _values(array):
    """Flatten a range to its numeric values, or return the error code it carries."""
    data = _numerics(array, to_number=coerce_to_number)
    if isinstance(data, str):
        return data
    return [float(x) for x in data]


def _percentile_inc(data: list[float], frac: float) -> float:
    """PERCENTILE.INC: linear interpolation between the two nearest ranks."""
    return float(np.percentile(np.asarray(data, dtype=float), frac * 100.0,
                               method="linear"))


@excel_helper()
def quartile(array, quart):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   quartile-function-93cf8f62-60cd-4fdb-8a92-8451041e1a2a
    # QUARTILE is the legacy name for QUARTILE.INC.
    data = _values(array)
    if isinstance(data, str):
        return data

    q = _scalar(quart)
    if isinstance(q, str):
        return q
    q = int(q)                      # Excel truncates a fractional quart toward zero

    if not data or q < 0 or q > 4:
        return NUM_ERROR
    return _percentile_inc(data, q / 4.0)


@excel_helper()
def quartile_inc(array, quart):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   quartile-inc-function-1bbacc80-5075-42f1-aed6-47d735c4819d
    return quartile(array, quart)


@excel_helper()
def median(*args):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   median-function-d0916313-4753-414c-8537-ce85bdd967d2
    data = _numerics(*args, to_number=coerce_to_number)
    if isinstance(data, str):
        return data
    if not data:
        return NUM_ERROR
    return _percentile_inc([float(x) for x in data], 0.5)


def _rank(number, ref, order, average_ties: bool):
    data = _values(ref)
    if isinstance(data, str):
        return data

    num = _scalar(number)
    if isinstance(num, str):
        return num
    ordr = _scalar(order, default=0)
    if isinstance(ordr, str):
        return ordr

    if not data:
        return NA_ERROR
    ties = sum(1 for v in data if v == num)
    if not ties:
        return NA_ERROR                         # Excel: the value must be in the range

    # order 0 (or omitted) ranks largest first; anything else ranks smallest first.
    if ordr == 0:
        better = sum(1 for v in data if v > num)
    else:
        better = sum(1 for v in data if v < num)
    best = better + 1
    return best + (ties - 1) / 2 if average_ties else best


@excel_helper()
def rank(number, ref, order=0):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   rank-function-6a2fc49d-1831-4a03-9d8c-c279cf99f723
    return _rank(number, ref, order, average_ties=False)


@excel_helper()
def rank_eq(number, ref, order=0):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   rank-eq-function-284858ce-8ef6-450e-b662-26245be04a40
    return _rank(number, ref, order, average_ties=False)


@excel_helper()
def rank_avg(number, ref, order=0):
    # Excel reference: https://support.microsoft.com/en-us/office/
    #   rank-avg-function-bd406a6f-eb38-4d73-aa8e-6d1c3c72e83a
    return _rank(number, ref, order, average_ties=True)
