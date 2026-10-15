"""Makes ``tests`` importable as a package.

The suite shares fixtures (``conftest.py``) and a couple of helpers; keeping this
an explicit package means ``import tests.support`` works from anywhere instead of
depending on pytest's rootdir insertion order.
"""
