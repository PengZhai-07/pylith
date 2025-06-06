// =================================================================================================
// This code is part of PyLith, developed through the Computational Infrastructure
// for Geodynamics (https://github.com/geodynamics/pylith).
//
// Copyright (c) 2010-2025, University of California, Davis and the PyLith Development Team.
// All rights reserved.
//
// See https://mit-license.org/ and LICENSE.md and for license information.
// =================================================================================================

/* Original mesh
 *
 * Cells are 0-1, vertices are 2-5.
 *
 *              3
 *             /|\
 *            / | \
 *           /  |  \
 *          /   |   \
 *         2    |    5
 *          \   |   /
 *           \  |  /
 *            \ | /
 *             \|/
 *              4
 *
 *
 * After adding cohesive elements
 *
 * Cells are 0-1, 2, vertices are 3-8.
 *
 *              7 -9- 4
 *             /|     |\
 *            / |     | \
 *           /  |     |  \
 *          /   |     |   \
 *         3    |     |    6
 *          \   |     |   /
 *           \  |     |  /
 *            \ |     | /
 *             \|     |/
 *              8-10- 5
 */

#include "CohesiveKinSrcsDataTri3.hh"

const char* pylith::faults::CohesiveKinSrcsDataTri3::_meshFilename =
    "data/tri3.mesh";

const int pylith::faults::CohesiveKinSrcsDataTri3::_spaceDim = 2;

const int pylith::faults::CohesiveKinSrcsDataTri3::_cellDim = 1;

const int pylith::faults::CohesiveKinSrcsDataTri3::_numBasis = 2;

const int pylith::faults::CohesiveKinSrcsDataTri3::_numQuadPts = 2;

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_quadPts[] = {
    -1.0, 1.0,
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_quadWts[] = {
    1.0, 1.0
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_basis[] = {
    1.0, 0.0,
    0.0, 1.0,
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_basisDeriv[] = {
    -0.5, 0.5,
    -0.5, 0.5,
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_verticesRef[] = {
    -1.0, 1.0
};

const int pylith::faults::CohesiveKinSrcsDataTri3::_id = 10;

const char* pylith::faults::CohesiveKinSrcsDataTri3::_label = "fault";

const char* pylith::faults::CohesiveKinSrcsDataTri3::_finalSlipFilename =
    "data/tri3_finalslip.spatialdb";

const char* pylith::faults::CohesiveKinSrcsDataTri3::_slipTimeFilename =
    "data/tri3_sliptime.spatialdb";

const char* pylith::faults::CohesiveKinSrcsDataTri3::_riseTimeFilename =
    "data/tri3_risetime.spatialdb";

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_fieldT[] = {
    8.1, 9.1,
    8.2, 9.2,
    8.3, 9.3,
    8.4, 9.4,
    8.5, 9.5,
    8.7, 9.7,
    8.6, 9.6, // 8
    8.8, 9.8, // 9
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_fieldIncr[] = {
    3.1, 4.1,
    3.2, 4.2, // 3
    3.3, 4.3, // 4
    3.4, 4.4,
    3.5, 4.5, // 6
    3.7, 4.7, // 7
    3.6, 4.6, // 15
    3.8, 4.8, // 16
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_jacobianLumped[] = {
    1.1, 1.1,
    1.2, 1.2, // 3
    1.3, 1.3, // 4
    1.4, 1.4,
    1.5, 1.5, // 6
    1.7, 1.7, // 7
    1.0, 1.0, // 15
    1.0, 1.0, // 16
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_orientation[] = {
    0.0, +1.0,  +1.0, 0.0,
    0.0, +1.0,  +1.0, 0.0
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_area[] = {
    1.0,
    1.0,
};

const int pylith::faults::CohesiveKinSrcsDataTri3::_numFaultVertices = 2;
const int pylith::faults::CohesiveKinSrcsDataTri3::_verticesFault[] = {
    2, 3
};
const int pylith::faults::CohesiveKinSrcsDataTri3::_edgesLagrange[] = {
    15, 16
};
const int pylith::faults::CohesiveKinSrcsDataTri3::_verticesNegative[] = {
    4, 5
};
const int pylith::faults::CohesiveKinSrcsDataTri3::_verticesPositive[] = {
    7, 8
};

const int pylith::faults::CohesiveKinSrcsDataTri3::_numCohesiveCells = 1;
const int pylith::faults::CohesiveKinSrcsDataTri3::_cellMappingFault[] = {
    4
};
const int pylith::faults::CohesiveKinSrcsDataTri3::_cellMappingCohesive[] = {
    2
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_residual[] = {
    0.0,  0.0,
    +8.6, +9.6, // 3
    +8.8, +9.8, // 4
    0.0,  0.0,
    -8.6, -9.6, // 6
    -8.8, -9.8, // 7
    -(8.5-8.2 - 0.08241148423 - 0.04322376757),
    -(9.5-9.2 - 1.89546413727 - 0.99414665414), // 15
    -(8.7-8.3 - 0.14794836271 - 0.05698088572),
    -(9.7-9.3 - 1.77538035254 - 0.68377062865), // 16
};

const PylithScalar pylith::faults::CohesiveKinSrcsDataTri3::_jacobian[] = {
    0.0, 0.0, // 2x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 2y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 3x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    -1.0, 0.0, // 8
    0.0, 0.0,
    0.0, 0.0, // 3y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0,-1.0, // 8
    0.0, 0.0,
    0.0, 0.0, // 4x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    -1.0, 0.0, //  9
    0.0, 0.0, // 4y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0,-1.0, // 9
    0.0, 0.0, // 5x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 5y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 6x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    +1.0, 0.0, // 8
    0.0, 0.0,
    0.0, 0.0, // 6y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0,+1.0, // 8
    0.0, 0.0,
    0.0, 0.0, // 7x
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    +1.0, 0.0, // 9
    0.0, 0.0, // 7y
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0,+1.0, // 9

    0.0, 0.0, // 8x
    -1.0, 0.0, // 3
    0.0, 0.0,
    0.0, 0.0,
    +1.0, 0.0, // 6
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 8y
    0.0,-1.0, // 3
    0.0, 0.0,
    0.0, 0.0,
    0.0,+1.0, // 6
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,

    0.0, 0.0, // 9x
    0.0, 0.0,
    -1.0, 0.0, //  4
    0.0, 0.0,
    0.0, 0.0,
    +1.0, 0.0, // 7
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0, // 9y
    0.0, 0.0,
    0.0,-1.0, // 4
    0.0, 0.0,
    0.0, 0.0,
    0.0,+1.0, // 7
    0.0, 0.0,
    0.0, 0.0,
};

pylith::faults::CohesiveKinSrcsDataTri3::CohesiveKinSrcsDataTri3(void) { // constructor
    meshFilename = const_cast<char*>(_meshFilename);
    spaceDim = _spaceDim;
    cellDim = _cellDim;
    numBasis = _numBasis;
    numQuadPts = _numQuadPts;
    quadPts = const_cast<PylithScalar*>(_quadPts);
    quadWts = const_cast<PylithScalar*>(_quadWts);
    basis = const_cast<PylithScalar*>(_basis);
    basisDeriv = const_cast<PylithScalar*>(_basisDeriv);
    verticesRef = const_cast<PylithScalar*>(_verticesRef);
    id = _id;
    label = const_cast<char*>(_label);
    finalSlipFilename = const_cast<char*>(_finalSlipFilename);
    slipTimeFilename = const_cast<char*>(_slipTimeFilename);
    riseTimeFilename = const_cast<char*>(_riseTimeFilename);
    fieldT = const_cast<PylithScalar*>(_fieldT);
    fieldIncr = const_cast<PylithScalar*>(_fieldIncr);
    jacobianLumped = const_cast<PylithScalar*>(_jacobianLumped);
    orientation = const_cast<PylithScalar*>(_orientation);
    area = const_cast<PylithScalar*>(_area);
    residual = const_cast<PylithScalar*>(_residual);
    jacobian = const_cast<PylithScalar*>(_jacobian);
    verticesFault = const_cast<int*>(_verticesFault);
    edgesLagrange = const_cast<int*>(_edgesLagrange);
    verticesNegative = const_cast<int*>(_verticesNegative);
    verticesPositive = const_cast<int*>(_verticesPositive);
    numFaultVertices = _numFaultVertices;
    cellMappingFault = const_cast<int*>(_cellMappingFault);
    cellMappingCohesive = const_cast<int*>(_cellMappingCohesive);
    numCohesiveCells = _numCohesiveCells;
} // constructor


pylith::faults::CohesiveKinSrcsDataTri3::~CohesiveKinSrcsDataTri3(void) {}


// End of file
