// =================================================================================================
// This code is part of PyLith, developed through the Computational Infrastructure
// for Geodynamics (https://github.com/geodynamics/pylith).
//
// Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
// All rights reserved.
//
// See https://mit-license.org/ and LICENSE.md and for license information.
// =================================================================================================

#include "DataWriterData.hh"

const int pylith::meshio::DataWriterData::DataWriterData::numVertexFields = 4;
const int pylith::meshio::DataWriterData::DataWriterData::numCellFields = 4;

// ----------------------------------------------------------------------
// Constructor
pylith::meshio::DataWriterData::DataWriterData(void) :
    meshFilename(0),
    faultLabel(0),
    faultId(0),
    bcLabel(0),
    timestepFilename(0),
    vertexFilename(0),
    cellFilename(0),
    time(0),
    timeFormat(0),
    cellsLabel(0),
    labelId(0),
    numVertices(0),
    vertexFieldsInfo(0),
    numCells(0),
    cellFieldsInfo(0) { // constructor
    for (int i = 0; i < numVertexFields; ++i) {
        vertexFields[i] = 0;
    } // for

    for (int i = 0; i < numCellFields; ++i) {
        cellFields[i] = 0;
    } // for
} // constructor


// ----------------------------------------------------------------------
// Destructor
pylith::meshio::DataWriterData::~DataWriterData(void) { // destructor
} // destructor


// End of file
