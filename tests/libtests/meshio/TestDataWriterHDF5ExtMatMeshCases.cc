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

#include "TestDataWriterHDF5ExtMatMeshCases.hh" // Implementation of class methods

#include "data/DataWriterHDF5DataMatMeshTri3.hh"
#include "data/DataWriterHDF5DataMatMeshQuad4.hh"
#include "data/DataWriterHDF5DataMatMeshTet4.hh"
#include "data/DataWriterHDF5DataMatMeshHex8.hh"

#include "pylith/utils/error.h" // USES PYLITH_METHOD_BEGIN/END

// ----------------------------------------------------------------------
CPPUNIT_TEST_SUITE_REGISTRATION(pylith::meshio::TestDataWriterHDF5ExtMatMeshTri3);
CPPUNIT_TEST_SUITE_REGISTRATION(pylith::meshio::TestDataWriterHDF5ExtMatMeshQuad4);
CPPUNIT_TEST_SUITE_REGISTRATION(pylith::meshio::TestDataWriterHDF5ExtMatMeshTet4);
CPPUNIT_TEST_SUITE_REGISTRATION(pylith::meshio::TestDataWriterHDF5ExtMatMeshHex8);

// ----------------------------------------------------------------------
// Setup testing data.
void
pylith::meshio::TestDataWriterHDF5ExtMatMeshTri3::setUp(void) { // setUp
    PYLITH_METHOD_BEGIN;

    TestDataWriterHDF5ExtMesh::setUp();
    _data = new DataWriterHDF5DataMatMeshTri3;
    _initialize();

    PYLITH_METHOD_END;
} // setUp


// ----------------------------------------------------------------------
// Setup testing data.
void
pylith::meshio::TestDataWriterHDF5ExtMatMeshQuad4::setUp(void) { // setUp
    PYLITH_METHOD_BEGIN;

    TestDataWriterHDF5ExtMesh::setUp();
    _data = new DataWriterHDF5DataMatMeshQuad4;
    _initialize();

    PYLITH_METHOD_END;
} // setUp


// ----------------------------------------------------------------------
// Setup testing data.
void
pylith::meshio::TestDataWriterHDF5ExtMatMeshTet4::setUp(void) { // setUp
    PYLITH_METHOD_BEGIN;

    TestDataWriterHDF5ExtMesh::setUp();
    _data = new DataWriterHDF5DataMatMeshTet4;
    _initialize();

    PYLITH_METHOD_END;
} // setUp


// ----------------------------------------------------------------------
// Setup testing data.
void
pylith::meshio::TestDataWriterHDF5ExtMatMeshHex8::setUp(void) { // setUp
    PYLITH_METHOD_BEGIN;

    TestDataWriterHDF5ExtMesh::setUp();
    _data = new DataWriterHDF5DataMatMeshHex8;
    _initialize();

    PYLITH_METHOD_END;
} // setUp


// End of file
