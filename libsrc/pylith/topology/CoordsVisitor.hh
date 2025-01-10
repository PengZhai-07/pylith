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

#include "pylith/topology/topologyfwd.hh" // forward declarations

#include "pylith/utils/petscfwd.h" // HASA PetscVec, PetscSection
#include "pylith/utils/arrayfwd.hh" // USES scalar_array

// CoordsVisitor ----------------------------------------------------------
/** @brief Helper class for accessing coordinates in a finite-element mesh.
 */
class pylith::topology::CoordsVisitor { // CoordsVisitor
    friend class TestCoordsVisitor; // unit testing

    // PUBLIC METHODS ///////////////////////////////////////////////////////
public:

    /** Default constructor (includes initialization).
     *
     * @param dmMesh PETSc DM for finite-element mesh.
     */
    CoordsVisitor(const PetscDM& dmMesh);

    /// Default destructor
    ~CoordsVisitor(void);

    /// Initialize cached data.
    void initialize(void);

    /// Clear cached data.
    void clear(void);

    /** Get the local coordinates array associated with the local PETSc Vec.
     *
     * @returns Local array.
     */
    PetscScalar* localArray(void) const;

    /** Get fiber dimension of coordinates for point.
     *
     * @param point Point in mesh.
     * @returns Fiber dimension.
     */
    PetscInt sectionDof(const PetscInt point) const;

    /** Get offset into coordinates array for point.
     *
     * @param point Point in mesh.
     * @returns Offset.
     */
    PetscInt sectionOffset(const PetscInt point) const;

    /** Get coordinates array associated with closure.
     *
     * @pre Must be followed by call to restoreClosure().
     *
     * @param coordsCell Array of coordinates for cell.
     * @param coordsSize Size of coordinates array.
     * @param cell Finite-element cell.
     */
    void getClosure(PetscScalar** coordsCell,
                    PetscInt* coordsSize,
                    const PetscInt cell) const;

    /** Get coordinates array associated with closure.
     *
     * @param coords Array of coordinates for cell.
     * @param cell Finite-element cell.
     */
    void getClosure(scalar_array* coordsCell,
                    const PetscInt cell) const;

    /** Restore coordinates array associated with closure.
     *
     * @pre Must be preceded by call to getClosure().
     *
     * @param coordsCell Array of coordinates for cell.
     * @param coordsSize Size of coordinates array.
     * @param cell Finite-element cell.
     */
    void restoreClosure(PetscScalar** coordsCell,
                        PetscInt* coordsSize,
                        const PetscInt cell) const;

    /** Optimize the closure operator by creating index for closures.
     *
     * @param dmMesh PETSc DM to optimize closure on coordinates field.
     */
    static
    void optimizeClosure(PetscDM dmMesh);

    // PRIVATE MEMBERS //////////////////////////////////////////////////////
private:

    const PetscDM _dm; ///< Cached PETSc dm for mesh.
    PetscSection _section; ///< Cached PETSc section.
    PetscVec _localVec; ///< Cached local PETSc Vec.
    PetscScalar* _localArray; ///< Cached local array.

    // NOT IMPLEMENTED //////////////////////////////////////////////////////
private:

    CoordsVisitor(const CoordsVisitor&); ///< Not implemented
    const CoordsVisitor& operator=(const CoordsVisitor&); ///< Not implemented

}; // CoordsVisitor

#include "CoordsVisitor.icc"

// End of file
