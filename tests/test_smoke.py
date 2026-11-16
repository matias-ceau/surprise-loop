import importlib


def test_package_imports() -> None:
    module = importlib.import_module("surprise_loop")
    assert module.__name__ == "surprise_loop"
