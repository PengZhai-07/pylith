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

/// Namespace for pylith package
namespace pylith {
    namespace meshio {
        class FieldFactory;
    } // meshio
} // pylith

#include "pylith/topology/topologyfwd.hh" // HOLDSA Field
#include "pylith/topology/FieldBase.hh" // USES FieldBase::Descretization

class pylith::meshio::FieldFactory {
    friend class TestFieldFactory; // unit testing

    // PUBLIC METHODS /////////////////////////////////////////////////////
public:

    /** Default constructor.
     *
     * @param[inout] fields Container with fields.
     */
    FieldFactory(pylith::topology::Field& field);

    /// Destructor.
    ~FieldFactory(void);

    /** Add scalar field and set field values.
     *
     * @param[in] discretization Discretization for scalar field.
     */
    void addScalar(const pylith::topology::FieldBase::Discretization& discretization);

    /** Add vector field.
     *
     * @param[in] discretization Discretization for vector field.
     */
    void addVector(const pylith::topology::FieldBase::Discretization& discretization);

    /** Add tensor field.
     *
     * @param[in] discretization Discretization for tensor field.
     */
    void addTensor(const pylith::topology::FieldBase::Discretization& discretization);

    /** Add other field.
     *
     * @param[in] discretization Discretization for other field.
     */
    void addOther(const pylith::topology::FieldBase::Discretization& discretization);

    /** Set values in field.
     *
     * @param[in] values Array of values for all subfields.
     * @param[in] numPoints Number of points associated with values.
     * @param[in] numDOF Total number of values per point.
     */
    void setValues(const PylithScalar* values,
                   const PylithInt numPoints,
                   const PylithInt numDOF);

    // PRIVATE MEMBERS ////////////////////////////////////////////////////
private:

    pylith::topology::Field& _field; ///< Field witn subfields.

    // NOT IMPLEMENTED ////////////////////////////////////////////////////
private:

    FieldFactory(const FieldFactory &); ///< Not implemented.
    const FieldFactory& operator=(const FieldFactory&); ///< Not implemented

}; // class FieldFactory

// End of file
