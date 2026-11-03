"""Import smoke tests for the initial scaffold."""

from importlib import import_module


def test_package_imports() -> None:
    package = import_module("ttc_operatorbench")

    assert package.__version__ == "0.1.0"


def test_subpackages_import() -> None:
    subpackages = [
        "core",
        "models",
        "tasks",
        "verifiers",
        "search",
        "schedulers",
        "evals",
        "logging",
        "systems",
    ]

    for subpackage in subpackages:
        import_module(f"ttc_operatorbench.{subpackage}")
