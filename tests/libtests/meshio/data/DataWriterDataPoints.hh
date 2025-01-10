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

// Data for testing writing interpolation of solution to points.

#include "DataWriterData.hh" // ISA DataWriterData

namespace pylith {
    namespace meshio {
        class DataWriterDataPoints;
    } // meshio
} // pylith

class pylith::meshio::DataWriterDataPoints : public DataWriterData { // DataWriterData
    // PUBLIC METHODS ///////////////////////////////////////////////////////
public:

    /// Constructor
    DataWriterDataPoints(void);

    /// Destructor
    virtual ~DataWriterDataPoints(void);

    // PUBLIC MEMBERS ///////////////////////////////////////////////////////
public:

    int numPoints; ///< Number of points for interpolation.
    int spaceDim; ///< Spatial dimension.
    PylithScalar* points; /// Points for interpolation.
    const char* const* names; ///< Station names for points.

}; // DataWriterData

// End of file
