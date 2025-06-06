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
# @file tests/fullscale/linearelasticity/nofaults-3d/sheartraction_gendb.py
#
# @brief Python script to generate spatial database with displacement
# boundary conditions for the shear test. The traction boundary
# conditions use UniformDB in the .cfg file.

import numpy


class GenerateDB(object):
    """Python object to generate spatial database with displacement
    boundary conditions for the shear test.
    """

    def __init__(self):
        """Constructor.
        """
        return

    def run(self):
        """Generate the database.
        """
        # Domain
        x = numpy.arange(-1.0e+4, 1.01e+4, 5.0e+3)
        y = numpy.arange(-1.0e+4, 1.01e+4, 5.0e+3)
        z = numpy.array([0])
        x3, y3, z3 = numpy.meshgrid(x, y, z)
        nptsX = x.shape[0]
        nptsY = y.shape[0]
        nptsZ = z.shape[0]

        xyz = numpy.zeros((nptsX * nptsY * nptsZ, 3), dtype=numpy.float64)
        xyz[:, 0] = x3.ravel()
        xyz[:, 1] = y3.ravel()
        xyz[:, 2] = z3.ravel()

        from sheartraction_soln import AnalyticalSoln
        soln = AnalyticalSoln()
        disp = soln.displacement(xyz)

        from spatialdata.geocoords.CSCart import CSCart
        cs = CSCart()
        cs.inventory.spaceDim = 3
        cs._configure()
        data = {
            "x": x,
            "y": y,
            "z": z,
            "points": xyz,
            "coordsys": cs,
            "data_dim": 2,
            "values": [
                {"name": "initial_amplitude_x",
                 "units": "m",
                 "data": numpy.ravel(disp[0, :, 0])},
                {"name": "initial_amplitude_y",
                 "units": "m",
                 "data": numpy.ravel(disp[0, :, 1])},
                {"name": "initial_amplitude_z",
                 "units": "m",
                 "data": numpy.ravel(disp[0, :, 2])},
            ]}

        from spatialdata.spatialdb.SimpleGridAscii import SimpleGridAscii
        io = SimpleGridAscii()
        io.inventory.filename = "sheartraction_disp.spatialdb"
        io._configure()
        io.write(data)
        return


# ======================================================================
if __name__ == "__main__":
    GenerateDB().run()


# End of file
