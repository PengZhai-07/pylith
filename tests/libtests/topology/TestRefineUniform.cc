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

#include "TestRefineUniform.hh" // Implementation of class methods

#include "pylith/topology/RefineUniform.hh" // USES RefineUniform
#include "tests/src/FaultCohesiveStub.hh" // USES FaultCohesiveStub

#include "pylith/topology/Mesh.hh" // USES Mesh
#include "pylith/topology/Stratum.hh" // USES Stratum
#include "pylith/topology/CoordsVisitor.hh" // USES CoordsVisitor
#include "pylith/meshio/MeshIOAscii.hh" // USES MeshIOAscii

#include "pylith/utils/array.hh" // USES int_array

#include <strings.h> // USES strcasecmp()
#include <stdexcept> // USES std::logic_error

#include "catch2/catch_test_macros.hpp"

// ------------------------------------------------------------------------------------------------
// Setup testing data.
pylith::topology::TestRefineUniform::TestRefineUniform(TestRefineUniform_Data* data) :
    _data(data) {
    assert(_data);
} // constructor


// ------------------------------------------------------------------------------------------------
// Destructor.
pylith::topology::TestRefineUniform::~TestRefineUniform(void) {
    delete _data;_data = NULL;
} // destructor


// ------------------------------------------------------------------------------------------------
// Test refine().
void
pylith::topology::TestRefineUniform::testRefine(void) {
    PYLITH_METHOD_BEGIN;
    assert(_data);

    Mesh mesh(_data->cellDim);
    _initializeMesh(&mesh);

    RefineUniform refiner;
    Mesh newMesh(_data->cellDim);
    refiner.refine(&newMesh, mesh, _data->refineLevel);

    // Check mesh dimension
    REQUIRE(_data->cellDim == newMesh.getDimension());

    const PetscDM& dmMesh = newMesh.getDM();assert(dmMesh);

    // Check vertices
    pylith::topology::Stratum verticesStratum(dmMesh, topology::Stratum::DEPTH, 0);
    REQUIRE(_data->numVertices == verticesStratum.size());
    const PetscInt vStart = verticesStratum.begin();
    const PetscInt vEnd = verticesStratum.end();

    pylith::topology::CoordsVisitor coordsVisitor(dmMesh);
    const int spaceDim = _data->spaceDim;
    for (PetscInt v = vStart; v < vEnd; ++v) {
        CHECK(spaceDim == coordsVisitor.sectionDof(v));
    } // for

    // Check cells
    pylith::topology::Stratum cellsStratum(dmMesh, topology::Stratum::HEIGHT, 0);
    const PetscInt cStart = cellsStratum.begin();
    const PetscInt cEnd = cellsStratum.end();
    const PetscInt numCells = cellsStratum.size();

    REQUIRE(_data->numCells+_data->numCellsCohesive == numCells);
    PetscErrorCode err;
    // Normal cells
    for (PetscInt c = cStart; c < _data->numCells; ++c) {
        DMPolytopeType ct;
        PetscInt *closure = NULL;
        PetscInt closureSize, numCorners = 0;

        err = DMPlexGetCellType(dmMesh, c, &ct);assert(!err);
        err = DMPlexGetTransitiveClosure(dmMesh, c, PETSC_TRUE, &closureSize, &closure);assert(!err);
        for (PetscInt p = 0; p < closureSize*2; p += 2) {
            const PetscInt point = closure[p];
            if ((point >= vStart) && (point < vEnd)) {
                closure[numCorners++] = point;
            } // if
        } // for
        err = DMPlexInvertCell(ct, closure);assert(!err);
        CHECK(_data->numCorners == numCorners);
        err = DMPlexRestoreTransitiveClosure(dmMesh, c, PETSC_TRUE, &closureSize, &closure);assert(!err);
    } // for

    // Cohesive cells
    for (PetscInt c = _data->numCells; c < cEnd; ++c) {
        DMPolytopeType ct;
        PetscInt *closure = NULL;
        PetscInt closureSize, numCorners = 0;

        err = DMPlexGetCellType(dmMesh, c, &ct);assert(!err);
        err = DMPlexGetTransitiveClosure(dmMesh, c, PETSC_TRUE, &closureSize, &closure);assert(!err);
        for (PetscInt p = 0; p < closureSize*2; p += 2) {
            const PetscInt point = closure[p];
            if ((point >= vStart) && (point < vEnd)) {
                closure[numCorners++] = point;
            } // if
        } // for
        err = DMPlexInvertCell(ct, closure);assert(!err);
        CHECK(_data->numCornersCohesive == numCorners);
        err = DMPlexRestoreTransitiveClosure(dmMesh, c, PETSC_TRUE, &closureSize, &closure);assert(!err);
    } // for

    // check materials
    PetscInt matId = 0;
    PetscInt matIdSum = 0; // Use sum of material ids as simple checksum.
    for (PetscInt c = cStart; c < cEnd; ++c) {
        err = DMGetLabelValue(dmMesh, pylith::topology::Mesh::cells_label_name, c, &matId);assert(!err);
        matIdSum += matId;
    } // for
    CHECK(_data->matIdSum == matIdSum);

    // Check groups
    PetscInt numGroups, pStart, pEnd;
    err = DMPlexGetChart(dmMesh, &pStart, &pEnd);assert(!err);
    err = DMGetNumLabels(dmMesh, &numGroups);assert(!err);
    for (PetscInt iGroup = 0; iGroup < _data->numGroups; ++iGroup) {
        // Omit depth, vtk, ghost and material-id labels
        // Don't know order of labels, so do brute force linear search
        bool foundLabel = false;
        int iLabel = 0;
        const char *name = NULL;
        PetscInt firstPoint = 0;

        while (iLabel < numGroups) {
            err = DMGetLabelName(dmMesh, iLabel, &name);assert(!err);
            if (0 == strcmp(_data->groupNames[iGroup], name)) {
                foundLabel = true;
                break;
            } else {
                ++iLabel;
            } // if/else
        } // while
        assert(foundLabel);

        for (PetscInt p = pStart; p < pEnd; ++p) {
            PetscInt val;
            err = DMGetLabelValue(dmMesh, name, p, &val);assert(!err);
            if (val >= 0) {
                firstPoint = p;
                break;
            } // if
        } // for
        std::string groupType = (firstPoint >= cStart && firstPoint < cEnd) ? "cell" : "vertex";
        CHECK(std::string(_data->groupTypes[iGroup]) == groupType);
        PetscInt numPoints;
        err = DMGetStratumSize(dmMesh, name, 1, &numPoints);assert(!err);
        REQUIRE(_data->groupSizes[iGroup] == numPoints);
        PetscIS pointIS = NULL;
        const PetscInt *points = NULL;
        err = DMGetStratumIS(dmMesh, name, 1, &pointIS);assert(!err);
        err = ISGetIndices(pointIS, &points);assert(!err);
        if (groupType == std::string("vertex")) {
            for (PetscInt p = 0; p < numPoints; ++p) {
                assert((points[p] >= 0 && points[p] < cStart) || (points[p] >= cEnd));
            } // for
        } else {
            for (PetscInt p = 0; p < numPoints; ++p) {
                assert(points[p] >= cStart && points[p] < cEnd);
            } // for
        } // if/else
        err = ISRestoreIndices(pointIS, &points);assert(!err);
        err = ISDestroy(&pointIS);assert(!err);
    } // for

    PYLITH_METHOD_END;
} // testRefine


// ------------------------------------------------------------------------------------------------
void
pylith::topology::TestRefineUniform::_initializeMesh(Mesh* const mesh) {
    PYLITH_METHOD_BEGIN;
    assert(_data);
    assert(mesh);

    pylith::meshio::MeshIOAscii iohandler;
    iohandler.setFilename(_data->filename);
    iohandler.read(mesh);

    // Adjust topology if necessary.
    if (_data->faultA) {
        faults::FaultCohesiveStub faultA;
        faultA.setCohesiveLabelValue(100);
        faultA.setSurfaceLabelName(_data->faultA);
        faultA.adjustTopology(mesh);
    } // if

    if (_data->faultB) {
        faults::FaultCohesiveStub faultB;
        faultB.setCohesiveLabelValue(101);
        faultB.setSurfaceLabelName(_data->faultB);
        faultB.adjustTopology(mesh);
    } // if

    PYLITH_METHOD_END;
} // _initializeMesh


// ------------------------------------------------------------------------------------------------
// Constructor
pylith::topology::TestRefineUniform_Data::TestRefineUniform_Data(void) :
    filename(NULL),
    refineLevel(0),
    faultA(NULL),
    faultB(NULL),
    isSimplexMesh(true),
    numVertices(0),
    spaceDim(0),
    cellDim(0),
    numCells(0),
    numCorners(0),
    numCellsCohesive(0),
    numCornersCohesive(0),
    matIdSum(0),
    groupSizes(NULL),
    groupNames(NULL),
    groupTypes(NULL),
    numGroups(0) { // constructor
} // constructor


// ------------------------------------------------------------------------------------------------
// Destructor
pylith::topology::TestRefineUniform_Data::~TestRefineUniform_Data(void) {} // destructor


// End of file
