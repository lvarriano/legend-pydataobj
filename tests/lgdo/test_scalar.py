import pytest

import pygama.lgdo as lgdo


def test_datatype_name():
    scalar = lgdo.Scalar(value=42)
    assert scalar.datatype_name() == "real"


def test_form_datatype():
    scalar = lgdo.Scalar(value=42)
    assert scalar.form_datatype() == "real"


def test_init():
    attrs = {"attr1": 1}
    scalar = lgdo.Scalar(value=42, attrs=attrs)
    assert scalar.value == 42
    assert scalar.attrs == attrs | {"datatype": "real"}

    with pytest.raises(ValueError):
        lgdo.Scalar(value=42, attrs={"datatype": "string"})


def test_getattrs():
    scalar = lgdo.Scalar(value=42, attrs={"attr1": 1})
    assert scalar.getattrs() == {"attr1": 1}
    assert scalar.getattrs(True) == {"attr1": 1, "datatype": "real"}


def test_equality():
    assert lgdo.Scalar(value=42) == lgdo.Scalar(value=42)
