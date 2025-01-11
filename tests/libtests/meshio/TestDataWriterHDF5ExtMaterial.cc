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

#include "TestDataWriterHDF5ExtMaterial.hh" // Implementation of class methods

#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/topology/Field.hh" // USES Field
#include "pylith/meshio/DataWriterHDF5Ext.hh" // USES DataWriterHDF5Ext
#include "pylith/meshio/OutputSubfield.hh" // USES OutputSubfield
#include "pylith/utils/error.hh" // USES PYLITH_METHOD*

// ------------------------------------------------------------------------------------------------
// Setup testing data.
pylith::meshio::TestDataWriterHDF5ExtMaterial::TestDataWriterHDF5ExtMaterial(TestDataWriterHDF5ExtMaterial_Data* data) :
    _data(data) {
    TestDataWriterMaterial::_initialize();
}


// ------------------------------------------------------------------------------------------------
// Tear down testing data.
pylith::meshio::TestDataWriterHDF5ExtMaterial::~TestDataWriterHDF5ExtMaterial(void) {
    PYLITH_METHOD_BEGIN;

    delete _data;_data = nullptr;

    PYLITH_METHOD_END;
} // tearDown


// ------------------------------------------------------------------------------------------------
// Test open() and close()
void
pylith::meshio::TestDataWriterHDF5ExtMaterial::testOpenClose(void) {
    PYLITH_METHOD_BEGIN;
    assert(_materialMesh);
    assert(_data);

    DataWriterHDF5Ext writer;

    writer.filename(_data->opencloseFilename);

    const bool isInfo = false;
    writer.open(*_materialMesh, isInfo);
    writer.close();

    checkFile(_data->opencloseFilename);

    PYLITH_METHOD_END;
} // testOpenClose


// ------------------------------------------------------------------------------------------------
// Test writeVertexField.
void
pylith::meshio::TestDataWriterHDF5ExtMaterial::testWriteVertexField(void) {
    PYLITH_METHOD_BEGIN;
    assert(_domainMesh);
    assert(_materialMesh);
    assert(_data);

    DataWriterHDF5Ext writer;

    pylith::topology::Field vertexField(*_domainMesh);
    _createVertexField(&vertexField);

    writer.filename(_data->vertexFilename);

    const PylithScalar timeScale = 4.0;
    writer.setTimeScale(timeScale);
    const PylithScalar t = _data->time / timeScale;

    const bool isInfo = false;
    writer.open(*_materialMesh, isInfo);
    writer.openTimeStep(t, *_materialMesh);

    const pylith::string_vector& subfieldNames = vertexField.getSubfieldNames();
    const size_t numFields = subfieldNames.size();
    for (size_t i = 0; i < numFields; ++i) {
        OutputSubfield* subfield = OutputSubfield::create(vertexField, *_materialMesh, subfieldNames[i].c_str(), 1, 0);
        assert(subfield);
        subfield->project(vertexField.getOutputVector());
        writer.writeVertexField(t, *subfield);
        delete subfield;subfield = NULL;
    } // for
    writer.closeTimeStep();
    writer.close();

    checkFile(_data->vertexFilename);

    PYLITH_METHOD_END;
} // testWriteVertexField


// ------------------------------------------------------------------------------------------------
// Test writeCellField.
void
pylith::meshio::TestDataWriterHDF5ExtMaterial::testWriteCellField(void) {
    PYLITH_METHOD_BEGIN;
    assert(_materialMesh);
    assert(_data);

    DataWriterHDF5Ext writer;

    pylith::topology::Field cellField(*_materialMesh);
    _createCellField(&cellField);

    writer.filename(_data->cellFilename);

    const PylithScalar timeScale = 4.0;
    writer.setTimeScale(timeScale);
    const PylithScalar t = _data->time / timeScale;

    const bool isInfo = false;
    writer.open(*_materialMesh, isInfo);
    writer.openTimeStep(t, *_materialMesh);

    const pylith::string_vector& subfieldNames = cellField.getSubfieldNames();
    const size_t numFields = subfieldNames.size();
    for (size_t i = 0; i < numFields; ++i) {
        OutputSubfield* subfield = OutputSubfield::create(cellField, *_materialMesh, subfieldNames[i].c_str(), 0, 0);
        assert(subfield);
        subfield->project(cellField.getOutputVector());
        writer.writeCellField(t, *subfield);
        delete subfield;subfield = NULL;
    } // for
    writer.closeTimeStep();
    writer.close();

    checkFile(_data->cellFilename);

    PYLITH_METHOD_END;
} // testWriteCellField


// ------------------------------------------------------------------------------------------------
// Get test data.
pylith::meshio::TestDataWriterMaterial_Data*
pylith::meshio::TestDataWriterHDF5ExtMaterial::_getData(void) {
    return _data;
} // _getData


// End of file
