# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================
# @file pylith/materials/AuxSubfieldsAbsorbingDampers.py
#
# @brief Python container for absorbing dampers subfields.

from pylith.utils.PetscComponent import PetscComponent


class AuxSubfieldsAbsorbingDampers(PetscComponent):
    """
    Auxiliary subfields for the absorbing dampers boundary condition.
    """
    DOC_CONFIG = {
        "cfg": """
            [absorbing_dampers_auxiliary_subfields]
            density.basis_order = 0
            vp.basis_order = 0
            vs.basis_order = 0            
            """,
    }

    import pythia.pyre.inventory

    from pylith.topology.Subfield import Subfield

    density = pythia.pyre.inventory.facility("density", family="auxiliary_subfield", factory=Subfield)
    density.meta['tip'] = "Mass density subfield."

    vs = pythia.pyre.inventory.facility("vs", family="auxiliary_subfield", factory=Subfield)
    vs.meta['tip'] = "Shear (S) wave speed subfield."

    vp = pythia.pyre.inventory.facility("vp", family="auxiliary_subfield", factory=Subfield)
    vp.meta['tip'] = "Dilatational (P) wave speed subfield."

    # PUBLIC METHODS /////////////////////////////////////////////////////

    def __init__(self, name="auxsubfieldsabsorbingdampers"):
        """Constructor.
        """
        PetscComponent.__init__(self, name, facility="auxiliary_subfields")
        return

    # PRIVATE METHODS ////////////////////////////////////////////////////

    def _configure(self):
        PetscComponent._configure(self)
        return


# FACTORIES ////////////////////////////////////////////////////////////

def auxiliary_subfields():
    """Factory associated with AuxSubfieldsAbsorbingDampers.
    """
    return AuxSubfieldsAbsorbingDampers()


# End of file
