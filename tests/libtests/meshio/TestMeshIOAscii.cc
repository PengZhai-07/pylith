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

#include "TestMeshIOAscii.hh" // Implementation of class methods

#include "pylith/meshio/MeshIOAscii.hh"
#include "pylith/topology/Mesh.hh" // USES Mesh

#include "pylith/utils/error.hh" // USES PYLITH_METHOD_BEGIN/END
#include "pylith/utils/journals.hh" // USES JournalingComponent

#include "catch2/catch_test_macros.hpp"

#include <strings.h> // USES strcasecmp()
#include <cassert> // USES assert()

// ----------------------------------------------------------------------
// Constructor.
pylith::meshio::TestMeshIOAscii::TestMeshIOAscii(TestMeshIO_Data* data) :
    TestMeshIO(data) {
    _io = new MeshIOAscii();assert(_io);
    _io->PyreComponent::setIdentifier("TestMeshIOAscii");
} // constructor


// ----------------------------------------------------------------------
// Destructor.
pylith::meshio::TestMeshIOAscii::~TestMeshIOAscii(void) {
    delete _io;_io = nullptr;
} // destructor


// ----------------------------------------------------------------------
// Test filename()
void
pylith::meshio::TestMeshIOAscii::testFilename(void) {
    PYLITH_METHOD_BEGIN;
    assert(_io);

    const std::string& filename = "hi.txt";
    _io->setFilename(filename.c_str());
    REQUIRE(filename == std::string(_io->getFilename()));

    PYLITH_METHOD_END;
} // testFilename


// ----------------------------------------------------------------------
// Test write() and read().
void
pylith::meshio::TestMeshIOAscii::testWriteRead(void) {
    PYLITH_METHOD_BEGIN;
    assert(_io);

    TestMeshIO::_createMesh();assert(_mesh);

    // Write mesh
    _io->setFilename(_data->filename.c_str());
    _io->write(_mesh);

    // Read mesh
    delete _mesh;_mesh = new pylith::topology::Mesh;
    _io->read(_mesh);

    // Make sure meshIn matches data
    TestMeshIO::_checkVals();

    PYLITH_METHOD_END;
} // testWriteRead


// ----------------------------------------------------------------------
// Test read().
void
pylith::meshio::TestMeshIOAscii::testRead(void) {
    PYLITH_METHOD_BEGIN;
    assert(_io);

    // Read mesh
    delete _mesh;_mesh = new pylith::topology::Mesh;
    _io->setFilename(_data->filename.c_str());
    _io->read(_mesh);

    // Make sure mesh matches data
    _checkVals();

    PYLITH_METHOD_END;
} // testRead


// ----------------------------------------------------------------------
// Test read() with error.
void
pylith::meshio::TestMeshIOAscii::testReadError(void) {
    PYLITH_METHOD_BEGIN;
    assert(_io);

    // Read mesh
    delete _mesh;_mesh = new pylith::topology::Mesh;
    _io->setFilename(_data->filename.c_str());
    CHECK_THROWS_AS(_io->read(_mesh), std::runtime_error);

    PYLITH_METHOD_END;
} // testRead


// End of file
