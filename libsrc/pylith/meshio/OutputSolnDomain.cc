// =================================================================================================
// This code is part of PyLith, developed through the Computational Infrastructure
// for Geodynamics (https://github.com/geodynamics/pylith).
//
// Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
// All rights reserved.
//
// See https://mit-license.org/ and LICENSE.md and for license information.
// =================================================================================================

#include <portinfo>

#include "pylith/meshio/OutputSolnDomain.hh" // implementation of class methods

#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/topology/Field.hh" // USES Field
#include "pylith/topology/FieldOps.hh" // USES FieldOps
#include "pylith/meshio/OutputSubfield.hh" // USES OutputSubfield

#include "pylith/utils/error.hh" // USES PYLITH_METHOD_*
#include "pylith/utils/journals.hh" // USES PYLITH_COMPONENT_*

// ---------------------------------------------------------------------------------------------------------------------
// Constructor
pylith::meshio::OutputSolnDomain::OutputSolnDomain(void) {
    PyreComponent::setName("outputsolndomain");
} // constructor


// ---------------------------------------------------------------------------------------------------------------------
// Destructor
pylith::meshio::OutputSolnDomain::~OutputSolnDomain(void) {}


// ---------------------------------------------------------------------------------------------------------------------
// Write data for step in solution.
void
pylith::meshio::OutputSolnDomain::_writeSolnStep(const PylithReal t,
                                                 const PylithInt tindex,
                                                 const pylith::topology::Field& solution) {
    PYLITH_METHOD_BEGIN;
    PYLITH_COMPONENT_DEBUG("_writeSolnStep(t="<<t<<", tindex="<<tindex<<", solution="<<solution.getLabel()<<")");

    const pylith::string_vector& subfieldNames = pylith::topology::FieldOps::getSubfieldNamesDomain(solution);
    PetscVec solutionVector = solution.getOutputVector();assert(solutionVector);

    const size_t numSubfieldNames = subfieldNames.size();
    for (size_t iField = 0; iField < numSubfieldNames; iField++) {
        assert(solution.hasSubfield(subfieldNames[iField].c_str()));

        OutputSubfield* subfield = NULL;
        subfield = OutputObserver::_getSubfield(solution, solution.getMesh(), subfieldNames[iField].c_str());assert(subfield);
        subfield->project(solutionVector);

        if (0 == iField) {
            // Need output mesh from subfield (which may be refined).
            assert(subfield);
            pylith::topology::Mesh* outputMesh = _getOutputMesh(*subfield);
            _openSolnStep(t, *outputMesh);
        } // if
        OutputObserver::_appendField(t, *subfield);
    } // for
    _closeSolnStep();

    PYLITH_METHOD_END;
} // _writeSolnStep


// End of file
