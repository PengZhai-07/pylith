# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================
# @file tests/fullscale/linearelasticity/nofaults/2d/meshes.py
#
# @brief Mesh information for test cases.

from pylith.testing.FullTestApp import MeshEntity

class Tri(object):
    """Mesh information for tri mesh.
    """
    ENTITIES = {
        "domain": MeshEntity(ncells=128, ncorners=3, nvertices=81),
        "points": MeshEntity(ncells=3, ncorners=1, nvertices=3),

        # Materials
        "elastic_xpos": MeshEntity(ncells=64, ncorners=3, nvertices=45),
        "elastic_xneg": MeshEntity(ncells=64, ncorners=3, nvertices=45),

        # Boundaries
        "bc_xneg": MeshEntity(ncells=8, ncorners=2, nvertices=9),
        "bc_xpos": MeshEntity(ncells=8, ncorners=2, nvertices=9),
        "bc_yneg": MeshEntity(ncells=8, ncorners=2, nvertices=9),
        "bc_ypos": MeshEntity(ncells=8, ncorners=2, nvertices=9),
    }


class Quad(object):
    """Mesh information for tri mesh.
    """
    ENTITIES = {
        "domain": MeshEntity(ncells=90, ncorners=4, nvertices=110),
        "points": MeshEntity(ncells=3, ncorners=1, nvertices=3),

        "elastic_xpos": MeshEntity(ncells=45, ncorners=4, nvertices=60),
        "elastic_xneg": MeshEntity(ncells=45, ncorners=4, nvertices=60),

        "bc_xneg": MeshEntity(ncells=9, ncorners=2, nvertices=10),
        "bc_xpos": MeshEntity(ncells=9, ncorners=2, nvertices=10),
        "bc_yneg": MeshEntity(ncells=10, ncorners=2, nvertices=11),
        "bc_ypos": MeshEntity(ncells=10, ncorners=2, nvertices=11),
    }



# End of file
