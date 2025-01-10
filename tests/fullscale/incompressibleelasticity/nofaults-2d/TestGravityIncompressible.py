#!/usr/bin/env nemesis
# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================
# @file tests/fullscale/linearelasticity/nofaults-2d/TestGravityIncompressible.py
#
# @brief Test suite for testing pylith with 2-D gravitational body forces for incompssible elasticity.

import unittest

from pylith.testing.FullTestApp import (FullTestCase, Check, check_data)

import meshes
import gravity_incompressible_soln


# -------------------------------------------------------------------------------------------------
class TestCase(FullTestCase):

    def setUp(self):
        defaults = {
            "filename": "output/{name}-{mesh_entity}.h5",
            "exact_soln": gravity_incompressible_soln.AnalyticalSoln(),
            "mesh": self.mesh,
        }
        self.checks = [
            Check(
                mesh_entities=["domain", "points"],
                vertex_fields=["displacement", "pressure"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["elastic_xpos", "elastic_xneg"],
                filename="output/{name}-{mesh_entity}_info.h5",
                cell_fields = ["density", "bulk_modulus", "shear_modulus",  "gravitational_acceleration"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["elastic_xpos", "elastic_xneg"],
                vertex_fields = ["displacement", "pressure"],
                cell_fields = ["cauchy_strain", "cauchy_stress"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["bc_xneg", "bc_xpos", "bc_yneg", "bc_ypos"],
                filename="output/{name}-{mesh_entity}_info.h5",
                cell_fields=["initial_amplitude"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["bc_xneg", "bc_xpos", "bc_yneg"],
                vertex_fields=["displacement", "pressure"],
                defaults=defaults,
            ),
        ]

    def run_pylith(self, testName, args):
        FullTestCase.run_pylith(self, testName, args)

# -------------------------------------------------------------------------------------------------
class TestQuad(TestCase):

    def setUp(self):
        self.name = "gravity_incompressible_quad"
        self.mesh = meshes.Quad()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["gravity_incompressible.cfg", "gravity_incompressible_quad.cfg"])
        return


# -------------------------------------------------------------------------------------------------
class TestTri(TestCase):

    def setUp(self):
        self.name = "gravity_incompressible_tri"
        self.mesh = meshes.Tri()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["gravity_incompressible.cfg", "gravity_incompressible_tri.cfg"])
        return


# -------------------------------------------------------------------------------------------------
class TestQuadIC(TestCase):

    def setUp(self):
        self.name = "gravity_incompressible_ic_quad"
        self.mesh = meshes.Quad()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["gravity_incompressible_ic.cfg", "gravity_incompressible_ic_quad.cfg"])
        return


# -------------------------------------------------------------------------------------------------
class TestTriIC(TestCase):

    def setUp(self):
        self.name = "gravity_incompressible_ic_tri"
        self.mesh = meshes.Tri()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["gravity_incompressible_ic.cfg", "gravity_incompressible_ic_tri.cfg"])
        return


# ------------------------------------------------------------------------------------------------------
def test_cases():
    return [
        TestQuad,
        TestTri,
        TestQuadIC,
        TestTriIC,
    ]


# ------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    FullTestCase.parse_args()

    suite = unittest.TestSuite()
    for test in test_cases():
        suite.addTest(unittest.makeSuite(test))
    unittest.TextTestRunner(verbosity=2).run(suite)


# End of file
