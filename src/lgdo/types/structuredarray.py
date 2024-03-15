"""
Implements a LEGEND Data Object representing a special type of structured array.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from warnings import warn

import awkward as ak
import numexpr as ne
import numpy as np
from numpy.lib import recfunctions
import pandas as pd
from pandas.io.formats import format as fmt
import json

from .array import Array
from .arrayofequalsizedarrays import ArrayOfEqualSizedArrays
from .lgdo import LGDO
# from .scalar import Scalar
# from .struct import Struct
# from .vectorofvectors import VectorOfVectors
from .table import Table

log = logging.getLogger(__name__)

class StructuredArray(LGDO):
    """A special type based on numpy's structured arrays. Structured arrays are ndarrays whose datatype is a 
    composition of simpler datatypes organized as a sequence of named fields. However, note that the 
    :class:`.StructuredArray` is written to disk as a 2-D np.ndarray of fixed type. Fields will have their dtype 
    promoted to the smallest required dtype. 
    
    For this reason, it is discouraged to use a :class:`.StructuredArray` to
    hold a large amount of disparate data, such as one field that is a large array of `int8` and one field that is a 
    `float64`. In this example, when the :class:`.StructuredArray` is written to disk, all data will be promoted to 
    `float64`, which can take up a large amount of storage.

    https://numpy.org/doc/stable/user/basics.rec.html

    https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array


    """

    def __init__(
        self,
        nda: np.ndarray | None,
        attrs: dict[str, Any] = {},
    ) -> None:
        r"""
        Parameters
        ----------
        nda
            Instantiate the :class:`.StructuredArray` using the supplied numpy structured array. (Note: to instantiate 
            using a normal `numpy.ndarray`, use `StructuredArray.from_nda()` to instantiate.)
        attrs
            A set of user attributes to be carried along with this LGDO.
        """

        if not isinstance(nda, np.ndarray):
            msg = (
                f"To instantiate with `StructuredArray()`, you must
                pass a numpy structured array. To instantiate with a normal numpy ndarray, use 
                `StructedTable.from_nda()`. To instantiate with a `dict` of `LGDO` objects, use 
                `StructuredArray.from_lgdo()`."
            )
            raise TypeError(msg)  

        if nda.dtype.names is None:
            msg = (
                f"passed array looks like a normal numpy ndarray. To instantiate with `StructuredArray()`, you must
                pass a numpy structured array. To instantiate with a normal numpy ndarray, use 
                `StructedTable.from_nda()`. To instantiate with a `dict` of `LGDO` objects, use 
                `StructuredArray.from_lgdo()`."
            )
            raise TypeError(msg)   
        
        self.nda = nda
        for key in attrs.keys():
            self.attrs[key] |= attrs[key]
        self.attrs["nda_dtype"] |= self.nda.dtype

    @classmethod
    def from_lgdo(
        cls, 
        obj_dict: dict[str, LGDO],
        attrs: dict[str, Any] = {},
    ):
        r"""
        Parameters
        ----------
        obj_dict
            Instantiate the StructuredArray using the supplied named array-like LGDO's.
            An error will be raised if all arrays do not have the same first dimension.
        attrs
            A set of user attributes to be carried along with this LGDO.
        """

        if obj_dict is None or len(obj_dict) < 0:
            msg = (
                f"You must pass a dictionary of LGDO objects to instantiate a :class:`.StructuredArray` 
                with `StructuredArray.from_lgdo()`. To instantiate with a numpy structured array, use `StructuredArray()`. 
                To instantiate with a normal numpy ndarray, use `StructedTable.from_nda()`."
            )
            raise ValueError(msg)  
              
        if attrs is not isinstance(attrs, dict):
            msg = (
                f" `attrs` must be a `dict` but was given `{type(attrs)}`."
            )
            raise ValueError(msg) 
        
        # check that inputs 1) exist and are of correct type and 2) have same first dimension (# rows).
        # get the rest of the information needed to construct the StructuredArray.
        numrows = 0
        nda_dtype = []
        metadata = {}
        for key, i in enumerate(obj_dict.keys()):
            if key in list(zip(*nda_dtype))[0]:
                msg = (
                    f"More than one array named '{key}' is not allowed."
                )
                raise ValueError(msg)
            
            obj = obj_dict[key]
            # more support could be added for other objects
            if not (isinstance(obj, Array) or isinstance(obj, ArrayOfEqualSizedArrays)):
                msg = (
                    f"Got type {type(obj)} but only :class:`.Array` and :class:`.ArrayOfEqualSizedArrays` 
                    are supported."
                )
                if isinstance(obj, np.ndarray):
                    msg = msg + (
                        f"To instantiate with a numpy structured array, use `StructuredArray()`. 
                        To instantiate with a normal numpy ndarray, use `StructedTable.from_nda()`."
                    )
                raise TypeError(msg)

            shape = obj.nda.shape

            if i == 0:
                numrows = shape[0]
            elif shape[0] != numrows:
                msg = (
                    f"First array has length {numrows} but other array has length {shape[0]}. All arrays must have 
                    same first dimension."
                )
                raise ValueError(msg)
        
            metadata |= {key: obj.attrs}

            # for numpy structured array
            # this is a list of tuples
            nda_dtype.append((key, obj.nda.dtype, shape[1:]))
    
        # now we know all inputs have same first dimension and we have collected the required number of rows
        # next step is to create a structured np.ndarray to hold the data.
        nda_dtype = np.dtype(nda_dtype)
        nda = np.empty(numrows, dtype=nda_dtype)
        
        # fill the structured array
        for key in obj_dict.keys():
            nda[key] = obj_dict[key].nda

        # keep some info, but note that it has to be written to file as a string
        # so do str(...).encode()
            
        # but the metadata is a dict of dicts, so maybe this needs to be converted to a JSON string?
        
        # https://stackoverflow.com/questions/988228/convert-a-string-representation-of-a-dictionary-to-a-dictionary
        # s = "{'muffin' : 'lolz', 'foo' : 'kitty'}"
        # json_acceptable_string = s.replace("'", "\"")
        # d = json.loads(json_acceptable_string)
            
        attrs["nda_dtype"] |= nda.dtype
        # self.attrs["fields"] = self.nda.dtype.names
        for key in metadata.keys():
            attrs[key] |= metadata[key]

        return cls(nda=nda, attrs=attrs)

    @classmethod
    def from_nda(
        cls,
        nda: np.ndarray | None,
        nda_dtype: tuple[tuple[str,str,tuple[int,...]],...],
        attrs: dict[str,Any] = {},
        casting: str = 'safe',
        ):
        r"""
        Creates a StructuredArray from an array and some additional information. Intended to be used when loading a
        :class:`.StructuredArray` from disk.
        """

        if not isinstance(nda, np.ndarray):
            msg = (
                f"Got type {type(nda)} but need type `np.ndarray`. To instantiate with a numpy structured array, 
                use `StructuredArray()`. To instantiate with a normal numpy ndarray, use `StructedTable.from_nda()`.
                To instantiate with a `dict` of `LGDO` objects, use `StructuredArray.from_lgdo()`."
            )
            raise ValueError(msg)              
        
        if nda.dtype.names is not None:
            msg = (
                f"passed array looks like a structured array. To instantiate with a numpy structured array, 
                use `StructuredArray()`. To instantiate with a normal numpy ndarray, use `StructedTable.from_nda()`."
            )
            raise TypeError(msg)  
    
        if nda is None:
            # make an empty (len == 0) structured array of the correct dtype
            # start with bool as lowest size so it can be promoted to other dtypes
            nda_sa = recfunctions.unstructured_to_structured(arr=np.empty(shape=(0,len(nda_dtype)), dtype=bool), 
                                                             dtype=nda_dtype, casting=casting)
        else:
            # I'm going to let numpy handle checking that the passed data and nda_dtype match appropriately...
            nda_sa = recfunctions.unstructured_to_structured(arr=nda, dtype=nda_dtype, casting=casting)

        return cls(nda=nda_sa, attrs=attrs)

    # done
    def datatype_name(self) -> str:
        return "StructuredArray"

    # done
    def __len__(self) -> int:
        """Provides ``__len__`` for this array-like class."""
        return self.nda.shape[0] # changed from self.shape[0]

    def explode(
            self, 
            table: Table = None,
            ) -> None | Table:
        r"""Explodes the :class:`.StructuredArray` into a :class:`.Table`. If a :class:`.Table` is provided, 
        returns `None` and exploded fields are added to the provided :class:`.Table`. If no :class:`.Table` is 
        provided, a new :class:`.Table` is created and the exploded fields are added to it and the new 
        :class:`.Table` is returned. """

        pass




    # def resize(self, new_size: int | None = None, do_warn: bool = False) -> None:
    #     # if new_size = None, use the size from the first field
    #     for field, obj in self.items():
    #         if new_size is None:
    #             new_size = len(obj)
    #         elif len(obj) != new_size:
    #             if do_warn:
    #                 log.warning(
    #                     f"warning: resizing field {field}"
    #                     f"with size {len(obj)} != {new_size}"
    #                 )
    #             if isinstance(obj, Table):
    #                 obj.resize(new_size)
    #             else:
    #                 obj.resize(new_size)
    #     self.size = new_size

    # def push_row(self) -> None:
    #     self.loc += 1

    # def is_full(self) -> bool:
    #     return self.loc >= self.size

    # def clear(self) -> None:
    #     self.loc = 0

    # def add_field(self, name: str, obj: LGDO, use_obj_size: bool = False) -> None:
    #     """Add a field (column) to the table.

    #     Use the name "field" here to match the terminology used in
    #     :class:`.Struct`.

    #     Parameters
    #     ----------
    #     name
    #         the name for the field in the table.
    #     obj
    #         the object to be added to the table.
    #     use_obj_size
    #         if ``True``, resize the table to match the length of `obj`.
    #     """
    #     if not hasattr(obj, "__len__"):
    #         msg = "cannot add field of type"
    #         raise TypeError(msg, type(obj).__name__)

    #     super().add_field(name, obj)

    #     if self.size is None:
    #         self.size = len(obj)

    #     # check / update sizes
    #     if self.size != len(obj):
    #         warn(
    #             f"warning: you are trying to add {name} with length {len(obj)} to a table with size {self.size} and data might be lost. \n"
    #             f"With 'use_obj_size' set to:\n"
    #             f"  - True, the table will be resized to length {len(obj)} by padding/clipping its columns.\n"
    #             f"  - False (default), object {name} will be padded/clipped to length {self.size}.",
    #             UserWarning,
    #             stacklevel=2,
    #         )
    #         new_size = len(obj) if use_obj_size else self.size
    #         self.resize(new_size=new_size)

    # def add_column(self, name: str, obj: LGDO, use_obj_size: bool = False) -> None:
    #     """Alias for :meth:`.add_field` using table terminology 'column'."""
    #     self.add_field(name, obj, use_obj_size=use_obj_size)

    # def remove_column(self, name: str, delete: bool = False) -> None:
    #     """Alias for :meth:`.remove_field` using table terminology 'column'."""
    #     super().remove_field(name, delete)

    # def eval(
    #     self,
    #     expr: str,
    #     parameters: Mapping[str, str] | None = None,
    # ) -> LGDO:
    #     """Apply column operations to the table and return a new LGDO.

    #     Internally uses :func:`numexpr.evaluate` if dealing with columns
    #     representable as NumPy arrays or :func:`eval` if
    #     :class:`.VectorOfVectors` are involved. In the latter case, the VoV
    #     columns are viewed as :class:`ak.Array` and the respective routines are
    #     therefore available.

    #     Parameters
    #     ----------
    #     expr
    #         if the expression only involves non-:class:`.VectorOfVectors`
    #         columns, the syntax is the one supported by
    #         :func:`numexpr.evaluate` (see `here
    #         <https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/index.html>`_
    #         for documentation). Note: because of internal limitations,
    #         reduction operations must appear the last in the stack. If at least
    #         one considered column is a :class:`.VectorOfVectors`, plain
    #         :func:`eval` is used and :class:`ak.Array` transforms can be used
    #         through the ``ak.`` prefix. (NumPy functions are analogously
    #         accessible through ``np.``). See also examples below.
    #     parameters
    #         a dictionary of function parameters. Passed to
    #         :func:`numexpr.evaluate`` as `local_dict` argument or to
    #         :func:`eval` as `locals` argument.

    #     Examples
    #     --------
    #     >>> import lgdo
    #     >>> tbl = lgdo.Table(
    #     ...   col_dict={
    #     ...     "a": lgdo.Array([1, 2, 3]),
    #     ...     "b": lgdo.VectorOfVectors([[5], [6, 7], [8, 9, 0]]),
    #     ...   }
    #     ... )
    #     >>> print(tbl.eval("a + b"))
    #     [[6],
    #      [8 9],
    #      [11 12  3],
    #     ]
    #     >>> print(tbl.eval("np.sum(a) + ak.sum(b)"))
    #     41
    #     """
    #     if parameters is None:
    #         parameters = {}

    #     # get the valid python variable names in the expression
    #     c = compile(expr, "0vbb is real!", "eval")

    #     # make a dictionary of low-level objects (numpy or awkward)
    #     # for later computation
    #     self_unwrap = {}
    #     has_ak = False
    #     for obj in c.co_names:
    #         if obj in self.keys():
    #             if isinstance(self[obj], VectorOfVectors):
    #                 self_unwrap[obj] = self[obj].view_as("ak", with_units=False)
    #                 has_ak = True
    #             else:
    #                 self_unwrap[obj] = self[obj].view_as("np", with_units=False)

    #     # use numexpr if we are only dealing with numpy data types
    #     if not has_ak:
    #         out_data = ne.evaluate(
    #             expr,
    #             local_dict=(self_unwrap | parameters),
    #         )

    #         # need to convert back to LGDO
    #         # np.evaluate should always return a numpy thing?
    #         if out_data.ndim == 0:
    #             return Scalar(out_data.item())
    #         if out_data.ndim == 1:
    #             return Array(out_data)
    #         if out_data.ndim == 2:
    #             return ArrayOfEqualSizedArrays(nda=out_data)

    #         msg = (
    #             f"evaluation resulted in {out_data.ndim}-dimensional data, "
    #             "I don't know which LGDO this corresponds to"
    #         )
    #         raise RuntimeError(msg)

    #     # resort to good ol' eval()
    #     globs = {"ak": ak, "np": np}
    #     out_data = eval(expr, globs, (self_unwrap | parameters))  # noqa: PGH001

    #     # need to convert back to LGDO
    #     if isinstance(out_data, ak.Array):
    #         if out_data.ndim == 1:
    #             return Array(out_data.to_numpy())
    #         return VectorOfVectors(out_data)

    #     if np.isscalar(out_data):
    #         return Scalar(out_data)

    #     msg = (
    #         f"evaluation resulted in a {type(out_data)} object, "
    #         "I don't know which LGDO this corresponds to"
    #     )
    #     raise RuntimeError(msg)

    def __str__(self):
        opts = fmt.get_dataframe_repr_params()
        opts["show_dimensions"] = False
        opts["index"] = False

        try:
            string = self.view_as("pd").to_string(**opts)
        except ValueError:
            string = "Cannot print Table with VectorOfVectors yet!"

        string += "\n"
        for k, v in self.items():
            attrs = v.getattrs()
            if attrs:
                string += f"\nwith attrs['{k}']={attrs}"

        attrs = self.getattrs()
        if attrs:
            string += f"\nwith attrs={attrs}"

        return string

    def view_as(
        self,
        library: str,
        with_units: bool = False,
        cols: list[str] | None = None,
        prefix: str = "",
    ) -> pd.DataFrame | np.NDArray | ak.Array:
        r"""View the Table data as a third-party format data structure.

        This is typically a zero-copy or nearly zero-copy operation.

        Supported third-party formats are:

        - ``pd``: returns a :class:`pandas.DataFrame`
        - ``ak``: returns an :class:`ak.Array` (record type)

        Notes
        -----
        Conversion to Awkward array only works when the key is a string.

        Parameters
        ----------
        library
            format of the returned data view.
        with_units
            forward physical units to the output data.
        cols
            a list of column names specifying the subset of the table's columns
            to be added to the data view structure.
        prefix
            The prefix to be added to the column names. Used when recursively
            getting the dataframe of a :class:`Table` inside this
            :class:`Table`.

        See Also
        --------
        .LGDO.view_as
        """
        if cols is None:
            cols = self.keys()

        if library == "pd":
            df = pd.DataFrame()

            for col in cols:
                data = self[col]

                if isinstance(data, Table):
                    log.debug(f"viewing Table {col=!r} recursively")

                    tmp_df = data.view_as(
                        "pd", with_units=with_units, prefix=f"{prefix}{col}_"
                    )
                    for k, v in tmp_df.items():
                        df[k] = v

                else:
                    log.debug(
                        f"viewing {type(data).__name__} column {col!r} as Pandas Series"
                    )
                    df[f"{prefix}{col}"] = data.view_as("pd", with_units=with_units)

            return df

        if library == "np":
            msg = f"Format {library!r} is not supported for Tables."
            raise TypeError(msg)

        if library == "ak":
            if with_units:
                msg = "Pint does not support Awkward yet, you must view the data with_units=False"
                raise ValueError(msg)

            # NOTE: passing the Table directly (which inherits from a dict)
            # makes it somehow really slow. Not sure why, but this could be due
            # to extra LGDO fields (like "attrs")
            return ak.Array({col: self[col].view_as("ak") for col in cols})

        msg = f"{library!r} is not a supported third-party format."
        raise TypeError(msg)
