// =================================================================================================
// This code is part of PyLith, developed through the Computational Infrastructure
// for Geodynamics (https://github.com/geodynamics/pylith).
//
// Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
// All rights reserved.
//
// See https://mit-license.org/ and LICENSE.md and for license information.
// =================================================================================================
#pragma once

#include "MeshData.hh"

namespace pylith {
    namespace meshio {
        class MeshData3DIndexOne;
    } // pylith
} // meshio

class pylith::meshio::MeshData3DIndexOne : public MeshData {
    // PUBLIC METHODS ///////////////////////////////////////////////////////
public:

    /// Constructor
    MeshData3DIndexOne(void);

    /// Destructor
    ~MeshData3DIndexOne(void);

    // PRIVATE MEMBERS //////////////////////////////////////////////////////
private:

    static const int _numVertices; ///< Number of vertices
    static const int _spaceDim; ///< Number of dimensions in vertex coordinates
    static const int _numCells; ///< Number of cells
    static const int _cellDim; ///< Number of dimensions associated with cell
    static const int _numCorners; ///< Number of vertices in cell

    static const PylithScalar _vertices[]; ///< Pointer to coordinates of vertices
    static const int _cells[]; ///< Pointer to indices of vertices in cells
    static const int _materialIds[]; ///< Pointer to cell material identifiers

    static const int _groups[]; ///< Groups of points
    static const int _groupSizes[]; ///< Sizes of groups
    static const char* _groupNames[]; ///< Array of group names
    static const char* _groupTypes[]; ///< Array of group types
    static const int _numGroups; ///< Number of groups

    static const bool _useIndexZero; ///< First vertex is 0 if true, 1 if false

};

// End of file
