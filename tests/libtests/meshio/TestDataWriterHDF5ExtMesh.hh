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

#include "TestDataWriterHDF5.hh" // ISA TestDataWriterHDF5
#include "TestDataWriterMesh.hh" // ISA TestDataWriterMesh

#include "pylith/topology/topologyfwd.hh" // USES Mesh, Field

namespace pylith {
    namespace meshio {
        class TestDataWriterHDF5ExtMesh;
        class TestDataWriterHDF5ExtMesh_Data;
    } // meshio
} // pylith

// ------------------------------------------------------------------------------------------------
class pylith::meshio::TestDataWriterHDF5ExtMesh : public TestDataWriterHDF5, public TestDataWriterMesh {
    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor.
    TestDataWriterHDF5ExtMesh(TestDataWriterHDF5ExtMesh_Data* data);

    /// Destructor.
    ~TestDataWriterHDF5ExtMesh(void);

    /// Test filename()
    static
    void testAccessors(void);

    /// Test hdf5Filename.
    static
    void testHdf5Filename(void);

    /// Test datasetFilename.
    static
    void testDatasetFilename(void);

    /// Test open() and close()
    void testOpenClose(void);

    /// Test writeVertexField.
    void testWriteVertexField(void);

    /// Test writeCellField.
    void testWriteCellField(void);

    // PROTECTED METHODS //////////////////////////////////////////////////////////////////////////
protected:

    /** Get test data.
     *
     * @returns Test data.
     */
    TestDataWriter_Data* _getData(void);

    // PROTECTED MEMBDERS /////////////////////////////////////////////////////////////////////////
protected:

    TestDataWriterHDF5ExtMesh_Data* _data; ///< Data for testing.

}; // class TestDataWriterHDF5ExtMesh

// ------------------------------------------------------------------------------------------------
class pylith::meshio::TestDataWriterHDF5ExtMesh_Data : public TestDataWriterHDF5_Data, public TestDataWriter_Data {};

// End of file
