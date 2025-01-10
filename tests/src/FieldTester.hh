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

#include "pylith/testing/testingfwd.hh" // forward declarations

#include "pylith/topology/Field.hh" // USES Field::SubfieldInfo
#include "pylith/utils/petscfwd.h" // USES PetscFE

#include "spatialdata/spatialdb/spatialdbfwd.hh" // USES SpatialDB

class pylith::testing::FieldTester {
    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /** Check to make sure field matches spatial database.
     *
     * @param[in] field Field to check.
     * @param[in] fieldDB Spatial database describing field.
     * @param[in] lengthScale Length scale for nondimensionalization.
     * @returns L2 norm of difference between field and spatial database.
     */
    static
    PylithReal checkFieldWithDB(const pylith::topology::Field& field,
                                spatialdata::spatialdb::SpatialDB* fieldDB,
                                const PylithReal lengthScale);

    /** Check that subfield information in field test subject matches expected subfield.
     *
     * @param field Field with subfields created by factory.
     * @param infoE Expected subfield info.
     */
    static
    void checkSubfieldInfo(const pylith::topology::Field& field,
                           const pylith::topology::Field::SubfieldInfo& infoE);

    // NOT IMPLEMENTED //////////////////////////////////////////////////////
private:

    FieldTester(void); ///< Not implemented.
    FieldTester(const FieldTester&); ///< Not implemented.
    const FieldTester& operator=(const FieldTester&); ///< Not implemented.

}; // FieldTester

// End of file
