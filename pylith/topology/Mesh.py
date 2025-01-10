# =================================================================================================
# This code is part of PyLith, developed through the Computational Infrastructure
# for Geodynamics (https://github.com/geodynamics/pylith).
#
# Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
# All rights reserved.
#
# See https://mit-license.org/ and LICENSE.md and for license information. 
# =================================================================================================

from .topology import Mesh as ModuleMesh


class Mesh(ModuleMesh):
    """
    Finite-element mesh defining the topology of the discretization.
    """

    def __init__(self, dim=None, comm=None, mesh=None, label=None, isSubmesh=False):
        """Constructor.
        """
        if comm is None and dim is None and label is None:
            ModuleMesh.__init__(self, isSubmesh)
        elif comm is None and mesh is None and label is None:
            ModuleMesh.__init__(self, dim)
        elif label is None:
            ModuleMesh.__init__(self, dim, comm.handle)
        else:
            assert(not mesh is None and not label is None)
            ModuleMesh.__init__(self, mesh, label)

        # Name of logging stage for mesh. We progress through various
        # stages as we read, distribute, and refine mesh.
        self.memLoggingStage = "Mesh"

    def comm(self):
        """Get communicator.
        """
        # Use Communicator object to wrap C++ MPI_Comm* returned by
        # module.
        from pylith.mpi.Communicator import Communicator
        return Communicator(ModuleMesh.getComm(self))

    def checkMaterialIds(self, materialIds):
        """Check material ids for consistency with mesh.
        """
        from .topology import MeshOps_checkMaterialIds
        MeshOps_checkMaterialIds(self, materialIds)

    def groupSizes(self):
        """Return the name and number of vertices for each group
        """
        groups = []
        names = ModuleMesh.groups(self)
        for name in names:
            groups.append((name, ModuleMesh.groupSize(self, name)))
        return groups

    def cleanup(self):
        """Deallocate locally managed data structures.
        """
        self.deallocate()


# End of file
