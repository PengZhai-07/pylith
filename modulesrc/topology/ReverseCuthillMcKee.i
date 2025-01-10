// =================================================================================================
// This code is part of PyLith, developed through the Computational Infrastructure
// for Geodynamics (https://github.com/geodynamics/pylith).
//
// Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
// All rights reserved.
//
// See https://mit-license.org/ and LICENSE.md and for license information.
// =================================================================================================

/**
 * @file modulesrc/topology/ReverseCuthillMcKee.hh
 *
 * @brief Python interface to C++ PyLith ReverseCuthillMcKee object.
 */

namespace pylith {
    namespace topology {
        // ReverseCuthillMcKee ----------------------------------------------
        class ReverseCuthillMcKee
        { // ReverseCuthillMcKee
          // PUBLIC METHODS /////////////////////////////////////////////////
public:

            /** Reorder vertices and cells of mesh using PETSc routines
             * implementing reverse Cuthill-McKee algorithm.
             *
             * @param mesh PyLith finite-element mesh.
             */
            static
            void reorder(topology::Mesh* mesh);

        }; // ReverseCuthillMcKee

    } // topology
} // pylith

// End of file
