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

#include "pylith/meshio/OutputSoln.hh" // ISA OutputSoln

#include "pylith/meshio/meshiofwd.hh"
#include "pylith/topology/topologyfwd.hh"
#include "pylith/utils/petscfwd.h"

#include "spatialdata/geocoords/geocoordsfwd.hh" // USES CoordSys

class pylith::meshio::OutputSolnPoints : public pylith::meshio::OutputSoln {
    friend class TestOutputSolnPoints; // unit testing

    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor.
    OutputSolnPoints(void);

    /// Destructor
    ~OutputSolnPoints(void);

    /// Deallocate PETSc and local data structures.
    void deallocate(void);

    /** Set coordinates and names of points.
     *
     * @param[in] points Array of coordinates [numPoints * spaceDim].
     * @param[in] numPoints Number of points.
     * @param[in] spaceDim Spatial dimension for coordinates.
     * @param[in] pointNames Array with point names.
     * @param[in] numPointNames Number of point banes.
     */
    void setPoints(const PylithReal* pointCoords,
                   const int numPoints,
                   const int spaceDim,
                   const char* const* pointNames,
                   const int numPointNames);

    // PROTECTED MEMBERS ///////////////////////////////////////////////////////////////////////////////////////////////
protected:

    /** Write solution at time step.
     *
     * @param[in] t Current time.
     * @param[in] tindex Current time step.
     * @param[in] solution Solution at time t.
     */
    void _writeSolnStep(const PylithReal t,
                        const PylithInt tindex,
                        const pylith::topology::Field& solution);

    /** Get output subfield, creating if necessary.
     *
     * @param[in] field Field containing subfields.
     * @param[in] submesh Submesh associated with output.
     * @param[in] name Name of subfield.
     */
    OutputSubfield* _getSubfield(const pylith::topology::Field& field,
                                 const pylith::topology::Mesh& submesh,
                                 const char* name);

    // PRIVATE MEMBERS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    /** Setup interpolatior.
     *
     * @param[in] solution Solution field.
     */
    void _setupInterpolator(const pylith::topology::Field& solution);

    /** Interpolate solution field.
     *
     * @param[in] solution Solution field to interpolate.
     */
    void _interpolateField(const pylith::topology::Field& solution);

    /// Write dataset with names of points to file.
    void _writePointNames(void);

    // PRIVATE MEMBERS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    pylith::scalar_array _pointCoords; ///< Array of point coordinates.
    pylith::string_vector _pointNames; ///< Array of point names.
    pylith::topology::Mesh* _pointMesh; ///< Mesh for points (no cells).
    pylith::topology::Field* _pointSoln; ///< Solution field at points.
    DMInterpolationInfo _interpolator; ///< Field interpolator.

    // NOT IMPLEMENTED /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    OutputSolnPoints(const OutputSolnPoints&); ///< Not implemented.
    const OutputSolnPoints& operator=(const OutputSolnPoints&); ///< Not implemented

}; // OutputSolnPoints

// End of file
