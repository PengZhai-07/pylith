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
 * @file libsrc/meshio/OutputPhysics.hh
 *
 * @brief Manager for output of finite-element data (e.g., material, boundary condition, interface condition).
 */
#pragma once

#include "meshiofwd.hh" // forward declarations

#include "pylith/problems/ObserverPhysics.hh" // ISA ObserverPhysics
#include "pylith/meshio/OutputObserver.hh" // ISA OutputObserver

#include "pylith/topology/topologyfwd.hh" // USES Field
#include "pylith/utils/array.hh" // HASA string_vector

class pylith::meshio::OutputPhysics :
    public pylith::problems::ObserverPhysics,
    public pylith::meshio::OutputObserver {
    friend class TestOutputPhysics; // unit testing

    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor
    OutputPhysics(void);

    /// Destructor
    virtual ~OutputPhysics(void);

    /// Deallocate PETSc and local data structures.
    virtual
    void deallocate(void) override;

    /** Set names of information fields requested for output.
     *
     * @param[in] names Array of field names.
     * @param[in] numNames Length of array.
     */
    void setInfoFields(const char* names[],
                       const int numNames);

    /** Get names of information fields requested for output.
     *
     * @returns Array of field names.
     */
    const pylith::string_vector& getInfoFields(void) const;

    /** Set names of data fields requested for output.
     *
     * @param[in] names Array of field names.
     * @param[in] numNames Length of array.
     */
    void setDataFields(const char* names[],
                       const int numNames);

    /** Get names of data fields requested for output.
     *
     * @returns Array of field names.
     */
    const pylith::string_vector& getDataFields(void) const;

    /** Set time scale.
     *
     * @param[in] value Time scale for dimensionalizing time.
     */
    void setTimeScale(const PylithReal value) override;

    /** Verify configuration.
     *
     * @param[in] solution Solution field.
     */
    void verifyConfiguration(const pylith::topology::Field& solution) const override;

    /** Receive update (subject of observer).
     *
     * @param[in] t Current time.
     * @param[in] tindex Current time step.
     * @param[in] solution Solution at time t.
     * @param[in] notification Type of notification.
     */
    void update(const PylithReal t,
                const PylithInt tindex,
                const pylith::topology::Field& solution,
                const NotificationType notification) override;

    // PROTECTED METHODS //////////////////////////////////////////////////////////////////////////
protected:

    /// Write diagnostic information.
    void _writeInfo(void);

    /** Prepare for output.
     *
     * @param[in] mesh Finite-element mesh object.
     * @param[in] isInfo True if only writing info values.
     */
    void _open(const pylith::topology::Mesh& mesh,
               const bool isInfo);

    /// Close output files.
    void _close(void);

    /** Prepare for output at this solution step.
     *
     * @param[in] t Time associated with field.
     * @param[in] mesh Mesh for output.
     */
    void _openDataStep(const PylithReal t,
                       const pylith::topology::Mesh& mesh);

    /// Finalize output at this solution step.
    void _closeDataStep(void);

    /** Write output for step in solution.
     *
     * @param[in] t Current time.
     * @param[in] tindex Current time step.
     * @param[in] solution Solution at time t.
     */
    void _writeDataStep(const PylithReal t,
                        const PylithInt tindex,
                        const pylith::topology::Field& solution);

    /** Names of information fields for output.
     *
     * Expand "all" into list of actual fields.
     *
     * @param[in] auxiliaryField Auxiliary field.
     * @param[in] diagnosticField Auxiliary field.
     */
    pylith::string_vector _expandInfoFieldNames(const pylith::topology::Field* auxiliaryField,
                                                const pylith::topology::Field* diagnosticField) const;

    /** Names of data fields for output.
     *
     * Expand "all" into list of actual fields in the solution field and derived field. We do not include subfields of
     * the auxiliary field, because most are physical parameters that do not change.
     *
     * @param[in] solution Solution field.
     * @param[in] auxiliaryField Auxiliary field.
     * @param[in] derivedField Derived field.
     */
    pylith::string_vector _expandDataFieldNames(const pylith::topology::Field& solution,
                                                const pylith::topology::Field* auxiliaryField,
                                                const pylith::topology::Field* derivedField) const;

    // PROTECTED MEMBERS //////////////////////////////////////////////////////////////////////////
protected:

    pylith::string_vector _infoFieldNames; ///< Names of subfields to output in info file.
    pylith::string_vector _dataFieldNames; ///< Names of subfields to output at time steps.

    // NOT IMPLEMENTED ////////////////////////////////////////////////////////////////////////////
private:

    OutputPhysics(const OutputPhysics&); ///< Not implemented.
    const OutputPhysics& operator=(const OutputPhysics&); ///< Not implemented

}; // OutputPhysics

// End of file
