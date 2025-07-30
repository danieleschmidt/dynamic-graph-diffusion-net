"""Test basic imports and package structure."""

import pytest


def test_package_import():
    """Test that the package can be imported."""
    import dgdn
    assert dgdn.__version__ == "0.1.0"


def test_package_metadata():
    """Test package metadata is accessible."""
    import dgdn
    assert hasattr(dgdn, "__version__")
    assert hasattr(dgdn, "__author__")
    assert hasattr(dgdn, "__email__")