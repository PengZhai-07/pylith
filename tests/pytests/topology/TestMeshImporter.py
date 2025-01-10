# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================

import unittest

from pylith.testing.TestCases import TestComponent, make_suite
from pylith.topology.MeshImporter import (MeshImporter, mesh_generator)


class TestMeshImporter(TestComponent):
    """Unit testing of MeshImporter object.
    """
    _class = MeshImporter
    _factory = mesh_generator


def load_tests(loader, tests, pattern):
    TEST_CLASSES = [TestMeshImporter]
    return make_suite(TEST_CLASSES, loader)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# End of file
