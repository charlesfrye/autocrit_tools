from collections import ChainMap
import itertools


class KwargTester(object):
    """Iterable of dictionaries suitable for use as kwargs, designed to test a function f
    by calling it with [f(**kwargs) for kwargs in kwarg_tester].

    The iterable is constructed from an iterable of co-dependent keyword argument names,
    called key_blocks, and an iterable (called val_blocks) of iterables (called val_sets)
    of desired combinations of values.

    The result is a product over val_sets organized as an iterable of dictionaries,
    all sharing the same keys and corresponding to different possible combinations of
    a val_set from each val_block.

    The simplest case is that each key_block is an iterable with one string, corresponding to a
    single keyword argument, and each val_block is an iterable containing one iterable with
    values corresponding to desired values for that keyword argument.
    The result is an iterable of dictionaries, corresponding to the set of all combinations of
    desired values for all keyword arguments.

    ```
    def f(foo, bar, baz=None):
        assert baz is None
        print(" ".join([bar]*foo))

    kwt = KwargTester([["foo", "bar"], ["baz"]], [[[1, "a"], [2, "b"]], [[None]]])

    [f(**kwargs) for kwargs in kwt];
    ```

    ```
    a
    b b
    ```
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.kwarg_product = self.make_kwarg_product()

    def make_kwarg_product(self):
        return itertools.product(*[block.dict_list for block in self.blocks])

    def flatten(self, kwargs_list):
        return dict(ChainMap(*kwargs_list))

    def __iter__(self):
        return [self.flatten(kwargs_list) for kwargs_list in self.kwarg_product].__iter__()

    @classmethod
    def from_raw(cls, keys_of_blocks, val_sets_of_blocks):
        blocks = [Block(keys, val_sets) for keys, val_sets
                  in zip(keys_of_blocks, val_sets_of_blocks)]
        return cls(blocks)


class Block(object):
    """A collection of keyword argument names and an iterable of iterables of
    desired combinations of values for those keyword arguments.

    These are combined into a list of dictionaries, block.dict_list.
    Each dictionary in the list has keys from self.keys and values from one
    of the val_sets.
    """

    def __init__(self, keys, val_sets):
        self.keys = keys
        self.val_sets = val_sets

    @property
    def dict_list(self):
        return [{key: val for key, val in zip(self.keys, val_set)} for val_set in self.val_sets]

    def __repr__(self):
        return (self.keys.__repr__(), self.val_sets.__repr__()).__repr__()
