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

#include "TestDataWriterHDF5ExtMesh.hh"

/// Namespace for pylith package
namespace pylith {
    namespace meshio {
        class TestDataWriterHDF5ExtMatMeshTri3;
        class TestDataWriterHDF5ExtMatMeshQuad4;
        class TestDataWriterHDF5ExtMatMeshTet4;
        class TestDataWriterHDF5ExtMatMeshHex8;
    } // meshio
} // pylith

// ----------------------------------------------------------------------
/// C++ unit testing for DataWriterHDF5Ext
class pylith::meshio::TestDataWriterHDF5ExtMatMeshTri3 : public TestDataWriterHDF5ExtMesh { // class
                                                                                            // TestDataWriterHDF5ExtMatMeshTri3
                                                                                            // CPPUNIT TEST SUITE
                                                                                            // /////////////////////////////////////////////////
    CPPUNIT_TEST_SUITE(TestDataWriterHDF5ExtMatMeshTri3);

    CPPUNIT_TEST(testOpenClose);
    CPPUNIT_TEST(testWriteVertexField);
    CPPUNIT_TEST(testWriteCellField);

    CPPUNIT_TEST_SUITE_END();

    // PUBLIC METHODS /////////////////////////////////////////////////////
public:

    /// Setup testing data.
    void setUp(void);

}; // class TestDataWriterHDF5ExtMatMeshTri3

// ----------------------------------------------------------------------
/// C++ unit testing for DataWriterHDF5Ext
class pylith::meshio::TestDataWriterHDF5ExtMatMeshQuad4 : public TestDataWriterHDF5ExtMesh { // class
                                                                                             // TestDataWriterHDF5ExtMatMeshQuad4
                                                                                             // CPPUNIT TEST SUITE
                                                                                             // /////////////////////////////////////////////////
    CPPUNIT_TEST_SUITE(TestDataWriterHDF5ExtMatMeshQuad4);

    CPPUNIT_TEST(testOpenClose);
    CPPUNIT_TEST(testWriteVertexField);
    CPPUNIT_TEST(testWriteCellField);

    CPPUNIT_TEST_SUITE_END();

    // PUBLIC METHODS /////////////////////////////////////////////////////
public:

    /// Setup testing data.
    void setUp(void);

}; // class TestDataWriterHDF5ExtMatMeshQuad4

// ----------------------------------------------------------------------
/// C++ unit testing for DataWriterHDF5Ext
class pylith::meshio::TestDataWriterHDF5ExtMatMeshTet4 : public TestDataWriterHDF5ExtMesh { // class
                                                                                            // TestDataWriterHDF5ExtMatMeshTet4
                                                                                            // CPPUNIT TEST SUITE
                                                                                            // /////////////////////////////////////////////////
    CPPUNIT_TEST_SUITE(TestDataWriterHDF5ExtMatMeshTet4);

    CPPUNIT_TEST(testOpenClose);
    CPPUNIT_TEST(testWriteVertexField);
    CPPUNIT_TEST(testWriteCellField);

    CPPUNIT_TEST_SUITE_END();

    // PUBLIC METHODS /////////////////////////////////////////////////////
public:

    /// Setup testing data.
    void setUp(void);

}; // class TestDataWriterHDF5ExtMatMeshTet4

// ----------------------------------------------------------------------
/// C++ unit testing for DataWriterHDF5Ext
class pylith::meshio::TestDataWriterHDF5ExtMatMeshHex8 : public TestDataWriterHDF5ExtMesh { // class
                                                                                            // TestDataWriterHDF5ExtMatMeshHex8
                                                                                            // CPPUNIT TEST SUITE
                                                                                            // /////////////////////////////////////////////////
    CPPUNIT_TEST_SUITE(TestDataWriterHDF5ExtMatMeshHex8);

    CPPUNIT_TEST(testOpenClose);
    CPPUNIT_TEST(testWriteVertexField);
    CPPUNIT_TEST(testWriteCellField);

    CPPUNIT_TEST_SUITE_END();

    // PUBLIC METHODS /////////////////////////////////////////////////////
public:

    /// Setup testing data.
    void setUp(void);

}; // class TestDataWriterHDF5ExtMatMeshHex8

// End of file
