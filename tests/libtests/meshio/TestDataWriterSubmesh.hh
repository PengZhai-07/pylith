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

#include "TestDataWriter.hh" // USES TestDataWriter_Data

#include "pylith/topology/topologyfwd.hh" // USES Mesh, Field

/// Namespace for pylith package
namespace pylith {
    namespace meshio {
        class TestDataWriterSubmesh;
        class TestDataWriterSubmesh_Data;
    } // meshio
} // pylith

// ------------------------------------------------------------------------------------------------
class pylith::meshio::TestDataWriterSubmesh {
    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor.
    TestDataWriterSubmesh(void);

    /// Destructor.
    ~TestDataWriterSubmesh(void);

    /// Set data for tri test case.
    static
    void setDataTri(TestDataWriterSubmesh_Data* data);

    /// Set data for quad test case.
    static
    void setDataQuad(TestDataWriterSubmesh_Data* data);

    /// Set data for tet test case.
    static
    void setDataTet(TestDataWriterSubmesh_Data* data);

    /// Set data for hex test case.
    static
    void setDataHex(TestDataWriterSubmesh_Data* data);

    // PROTECTED METHODS //////////////////////////////////////////////////////////////////////////
protected:

    /// Initialize mesh.
    void _initialize(void);

    /** Create vertex fields.
     *
     * @param fields Vertex fields.
     */
    void _createVertexField(pylith::topology::Field* field);

    /** Create cell fields.
     *
     * @param fields Cell fields.
     */
    void _createCellField(pylith::topology::Field* field);

    /** Get test data.
     *
     * @returns Test data.
     */
    virtual
    TestDataWriterSubmesh_Data* _getData(void) = 0;

    // PROTECTED MEMBERS //////////////////////////////////////////////////////////////////////////
protected:

    pylith::topology::Mesh* _mesh; ///< Mesh for domain
    pylith::topology::Mesh* _submesh; ///< Mesh for subdomain.

}; // class TestDataWriterSubmesh

// ------------------------------------------------------------------------------------------------
class pylith::meshio::TestDataWriterSubmesh_Data : public TestDataWriter_Data {
    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor
    TestDataWriterSubmesh_Data(void);

    /// Destructor
    ~TestDataWriterSubmesh_Data(void);

    // PUBLIC MEMBERS /////////////////////////////////////////////////////////////////////////////
public:

    const char* bcLabel; ///< Label marking submesh.

}; // class TestDataWriterSubmesh_Data

// End of file
