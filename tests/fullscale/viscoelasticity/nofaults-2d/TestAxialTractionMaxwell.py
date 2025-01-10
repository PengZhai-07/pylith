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
# @file tests/fullscale/viscoelasticity/nofaults-2d/TestAxialTractionMaxwell.py
#
# @brief Test suite for testing Maxwell material with 2-D axial extension (Neumann BC).

import unittest

from pylith.testing.FullTestApp import (FullTestCase, Check, check_data)

import meshes
import axialtraction_maxwell_soln
import axialtraction_maxwell_gendb


# -------------------------------------------------------------------------------------------------
class TestCase(FullTestCase):

    def setUp(self):
        defaults = {
            "filename": "output/{name}-{mesh_entity}.h5",
            "exact_soln": axialtraction_maxwell_soln.AnalyticalSoln(),
            "mesh": self.mesh,
        }
        self.checks = [
            Check(
                mesh_entities=["domain"],
                vertex_fields=["displacement"],
                defaults=defaults,
                tolerance=2.0e-5,
            ),
            Check(
                mesh_entities=["viscomat"],
                filename="output/{name}-{mesh_entity}_info.h5",
                cell_fields = ["density", "bulk_modulus", "shear_modulus", "maxwell_time"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["viscomat"],
                vertex_fields = ["displacement", "cauchy_strain", "cauchy_stress", "viscous_strain"],
                defaults=defaults,
                tolerance=2.0e-5,
            ),
            Check(
                mesh_entities=["bc_xneg", "bc_xpos", "bc_ypos", "bc_yneg"],
                filename="output/{name}-{mesh_entity}_info.h5",
                cell_fields=["initial_amplitude"],
                defaults=defaults,
            ),
            Check(
                mesh_entities=["bc_xneg", "bc_xpos", "bc_ypos", "bc_yneg"],
                vertex_fields=["displacement"],
                defaults=defaults,
                tolerance=2.0e-5,
            ),
        ]

    def run_pylith(self, testName, args):
        FullTestCase.run_pylith(self, testName, args, axialtraction_maxwell_gendb.GenerateDB)


# -------------------------------------------------------------------------------------------------
class TestQuad(TestCase):

    def setUp(self):
        self.name = "axialtraction_maxwell_quad"
        self.mesh = meshes.Quad()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["axialtraction_maxwell.cfg", "axialtraction_maxwell_quad.cfg"])
        return


# -------------------------------------------------------------------------------------------------
class TestTri(TestCase):

    def setUp(self):
        self.name = "axialtraction_maxwell_tri"
        self.mesh = meshes.Tri()
        super().setUp()

        TestCase.run_pylith(self, self.name, ["axialtraction_maxwell.cfg", "axialtraction_maxwell_tri.cfg"])
        return


# -------------------------------------------------------------------------------------------------
def test_cases():
    return [
        TestQuad,
        TestTri,
    ]


# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    FullTestCase.parse_args()

    suite = unittest.TestSuite()
    for test in test_cases():
        suite.addTest(unittest.makeSuite(test))
    unittest.TextTestRunner(verbosity=2).run(suite)


# End of file
