// -*- C++ -*-
//
// ======================================================================
//
//                           Brad T. Aagaard
//                        U.S. Geological Survey
//
// {LicenseText}
//
// ======================================================================
//

/* Original mesh
 *
 * Cells are 0-1, vertices are 2-6.
 *
 * 2   3,4,5  6
 *
 *     ^^^^^ Face in x-y plane
 *
 * After adding cohesive elements
 *
 * Cells are 0-1,10, vertices are 2-9.
 *
 * 2   3,4,5  7,8,19   6
 *             10,11,12
 *     ^^^^^^^^^^^^ Cohesive element in x-y plane.
 */

#include "CohesiveKinDataTet4.hh"

const char* pylith::faults::CohesiveKinDataTet4::_meshFilename =
  "data/tet4.mesh";

const int pylith::faults::CohesiveKinDataTet4::_spaceDim = 3;

const int pylith::faults::CohesiveKinDataTet4::_cellDim = 2;

const int pylith::faults::CohesiveKinDataTet4::_numBasis = 3;

const int pylith::faults::CohesiveKinDataTet4::_numQuadPts = 1;

const double pylith::faults::CohesiveKinDataTet4::_quadPts[] = {
  3.33333333e-01,  3.33333333e-01,
};

const double pylith::faults::CohesiveKinDataTet4::_quadWts[] = {
  5.00000000e-01,
};

const double pylith::faults::CohesiveKinDataTet4::_basis[] = {
  3.33333333e-01,  3.33333333e-01,
  3.33333333e-01,};

const double pylith::faults::CohesiveKinDataTet4::_basisDeriv[] = {
 -1.00000000e+00, -1.00000000e+00,
  1.00000000e+00,  0.00000000e+00,
  0.00000000e+00,  1.00000000e+00,
};

const double pylith::faults::CohesiveKinDataTet4::_verticesRef[] = {
 -1.00000000e+00, -1.00000000e+00,
  1.00000000e+00, -1.00000000e+00,
 -1.00000000e+00,  1.00000000e+00,
};

const int pylith::faults::CohesiveKinDataTet4::_id = 10;

const char* pylith::faults::CohesiveKinDataTet4::_label = "fault";

const char* pylith::faults::CohesiveKinDataTet4::_finalSlipFilename = 
  "data/tet4_finalslip.spatialdb";

const char* pylith::faults::CohesiveKinDataTet4::_slipTimeFilename = 
  "data/tet4_sliptime.spatialdb";

const char* pylith::faults::CohesiveKinDataTet4::_peakRateFilename = 
  "data/tet4_peakrate.spatialdb";

const char* pylith::faults::CohesiveKinDataTet4::_matPropsFilename = 
  "data/bulkprops_3d.spatialdb";

const double pylith::faults::CohesiveKinDataTet4::_fieldT[] = {
  7.1, 8.1, 9.1,
  7.2, 8.2, 9.2,
  7.3, 8.3, 9.3,
  7.4, 8.4, 9.4,
  7.5, 8.5, 9.5,
  7.6, 8.6, 9.6,
  7.8, 8.8, 9.8,
  7.0, 8.0, 9.0,
  7.7, 8.7, 9.7, // 10
  7.9, 8.9, 9.9, // 11
  7.1, 8.1, 9.1, // 12
};

const int pylith::faults::CohesiveKinDataTet4::_numConstraintVert = 3;

const double pylith::faults::CohesiveKinDataTet4::_orientation[] = {
  0.0, +1.0, 0.0,    0.0, 0.0, +1.0,    +1.0, 0.0, 0.0,
  0.0, +1.0, 0.0,    0.0, 0.0, +1.0,    +1.0, 0.0, 0.0,
  0.0, +1.0, 0.0,    0.0, 0.0, +1.0,    +1.0, 0.0, 0.0,
};

const int pylith::faults::CohesiveKinDataTet4::_constraintVertices[] = {
  10, 11, 12
};

const int pylith::faults::CohesiveKinDataTet4::_constraintCells[] = {
  15, 15, 15
};

const double pylith::faults::CohesiveKinDataTet4::_valsResidual[] = {
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  0.0,  0.0,  0.0,
  1.07974939836, -0.32861938211, 0.04694562602, // 10
  1.00381374723, -0.33460458241, 0.08365114560, // 11
  0.90493237602, -0.32577565537, 0.10859188512, // 12
};

const double pylith::faults::CohesiveKinDataTet4::_valsResidualIncr[] = {
  0.0,  0.0,  0.0,
  9.7,  7.7,  8.7, // 3
  9.9,  7.9,  8.9, // 4
  9.1,  7.1,  8.1, // 5
  0.0,  0.0,  0.0,
 -9.7, -7.7, -8.7, // 7
 -9.9, -7.9, -8.9, // 4
 -9.1, -7.1, -8.1, // 5
  0.01271057284, -0.00386843521, 0.00055263360, // 10
  0.01411467401, -0.00470489134, 0.00117622283, // 11
  0.01544188788, -0.00555907964, 0.00185302655, // 12
};

const double pylith::faults::CohesiveKinDataTet4::_valsJacobian[] = {
  0.0, 0.0, 0.0, // 2x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 2y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 2z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 3x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,-1.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 3y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 -1.0, 0.0, 0.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 3z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,-1.0, 0.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 4x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,-1.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 4y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 -1.0, 0.0, 0.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 4z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,-1.0, 0.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 5x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,-1.0, // 12
  0.0, 0.0, 0.0, // 5y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 -1.0, 0.0, 0.0, // 12
  0.0, 0.0, 0.0, // 5z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,-1.0, 0.0, // 12
  0.0, 0.0, 0.0, // 6x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 6y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 6z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 7x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 7y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 7z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 10
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 8x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 8y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 8z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 11
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 9x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 12
  0.0, 0.0, 0.0, // 9y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 12
  0.0, 0.0, 0.0, // 9z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 12
  0.0, 0.0, 0.0, // 10x
  0.0,-1.0, 0.0, // 3
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 7
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 10y
  0.0, 0.0,-1.0, // 3
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 7
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 10z
 -1.0, 0.0, 0.0, // 3
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 7
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 11x
  0.0, 0.0, 0.0,
  0.0,-1.0, 0.0, // 4
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 8
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 11y
  0.0, 0.0, 0.0,
  0.0, 0.0,-1.0, // 4
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 8
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 11z
  0.0, 0.0, 0.0,
 -1.0, 0.0, 0.0, // 4
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 8
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 12x
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,-1.0, 0.0, // 5
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0,+1.0, 0.0, // 9
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 12y
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,-1.0, // 5
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0,+1.0, // 9
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, // 12z
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 -1.0, 0.0, 0.0, // 5
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
 +1.0, 0.0, 0.0, // 9
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
  0.0, 0.0, 0.0,
};

const double pylith::faults::CohesiveKinDataTet4::_pseudoStiffness = 2.4;

pylith::faults::CohesiveKinDataTet4::CohesiveKinDataTet4(void)
{ // constructor
  meshFilename = const_cast<char*>(_meshFilename);
  spaceDim = _spaceDim;
  cellDim = _cellDim;
  numBasis = _numBasis;
  numQuadPts = _numQuadPts;
  quadPts = const_cast<double*>(_quadPts);
  quadWts = const_cast<double*>(_quadWts);
  basis = const_cast<double*>(_basis);
  basisDeriv = const_cast<double*>(_basisDeriv);
  verticesRef = const_cast<double*>(_verticesRef);
  id = _id;
  label = const_cast<char*>(_label);
  finalSlipFilename = const_cast<char*>(_finalSlipFilename);
  slipTimeFilename = const_cast<char*>(_slipTimeFilename);
  peakRateFilename = const_cast<char*>(_peakRateFilename);
  matPropsFilename = const_cast<char*>(_matPropsFilename);
  fieldT = const_cast<double*>(_fieldT);
  orientation = const_cast<double*>(_orientation);
  constraintVertices = const_cast<int*>(_constraintVertices);
  constraintCells = const_cast<int*>(_constraintCells);
  valsResidual = const_cast<double*>(_valsResidual);
  valsResidualIncr = const_cast<double*>(_valsResidualIncr);
  valsJacobian = const_cast<double*>(_valsJacobian);
  pseudoStiffness = _pseudoStiffness;
  numConstraintVert = _numConstraintVert;  
} // constructor

pylith::faults::CohesiveKinDataTet4::~CohesiveKinDataTet4(void)
{}


// End of file
