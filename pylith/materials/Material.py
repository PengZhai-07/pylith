# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================

from pylith.problems.Physics import Physics
from .materials import Material as ModuleMaterial


def validateDescription(value):
    """Validate description.
    """
    if 0 == len(value):
        raise ValueError("Description for material not specified.")
    return value


class Material(Physics, ModuleMaterial):
    """
    Abstract base class for a bulk material.
    """

    import pythia.pyre.inventory

    description = pythia.pyre.inventory.str("description", default="", validator=validateDescription)
    description.meta['tip'] = "Descriptive label for material."

    labelName = pythia.pyre.inventory.str("label", default="material-id", validator=pythia.pyre.inventory.choice(["material-id"]))
    labelName.meta['tip'] = "Name of label for material. Currently only 'material-id' is allowed."

    labelValue = pythia.pyre.inventory.int("label_value", default=1)
    labelValue.meta["tip"] = "Value of label for material."

    def __init__(self, name="material"):
        """Constructor.
        """
        Physics.__init__(self, name)

    def preinitialize(self, problem):
        """Setup material.
        """
        Physics.preinitialize(self, problem)
        ModuleMaterial.setDescription(self, self.description)
        ModuleMaterial.setLabelName(self, self.labelName)
        ModuleMaterial.setLabelValue(self, self.labelValue)


# End of file
