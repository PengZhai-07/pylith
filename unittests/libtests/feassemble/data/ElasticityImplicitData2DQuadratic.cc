// -*- C++ -*-
//
// ======================================================================
//
// Brad T. Aagaard, U.S. Geological Survey
// Charles A. Williams, GNS Science
// Matthew G. Knepley, University of Chicago
//
// This code was developed as part of the Computational Infrastructure
// for Geodynamics (http://geodynamics.org).
//
// Copyright (c) 2010 University of California, Davis
//
// See COPYING for license information.
//
// ======================================================================
//

// DO NOT EDIT THIS FILE
// This file was generated from python application elasticityapp.

#include "ElasticityImplicitData2DQuadratic.hh"

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_spaceDim = 2;

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_cellDim = 2;

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_numVertices = 6;

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_numCells = 1;

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_numBasis = 6;

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_numQuadPts = 6;

const char* pylith::feassemble::ElasticityImplicitData2DQuadratic::_matType = "ElasticPlaneStrain";

const char* pylith::feassemble::ElasticityImplicitData2DQuadratic::_matDBFilename = "data/elasticplanestrain.spatialdb";

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_matId = 0;

const char* pylith::feassemble::ElasticityImplicitData2DQuadratic::_matLabel = "elastic strain 2-D";

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_dt =   1.00000000e-02;

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_gravityVec[] = {
  0.00000000e+00, -1.00000000e+08,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_vertices[] = {
 -1.00000000e+00, -1.00000000e+00,
  1.00000000e+00,  2.00000000e-01,
 -1.50000000e+00,  5.00000000e-01,
 -2.50000000e-01,  3.50000000e-01,
 -1.25000000e+00, -2.50000000e-01,
  0.00000000e+00, -4.00000000e-01,
};

const int pylith::feassemble::ElasticityImplicitData2DQuadratic::_cells[] = {
0,1,2,3,4,5,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_verticesRef[] = {
 -1.00000000e+00, -1.00000000e+00,
  1.00000000e+00, -1.00000000e+00,
 -1.00000000e+00,  1.00000000e+00,
  0.00000000e+00,  0.00000000e+00,
 -1.00000000e+00,  0.00000000e+00,
  0.00000000e+00, -1.00000000e+00,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_quadPts[] = {
 -7.50000000e-01, -7.50000000e-01,
  7.50000000e-01, -7.50000000e-01,
 -7.50000000e-01,  7.50000000e-01,
  0.00000000e+00, -7.50000000e-01,
 -7.50000000e-01,  0.00000000e+00,
  2.50000000e-01,  2.50000000e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_quadWts[] = {
  3.33333333e-01,  3.33333333e-01,  3.33333333e-01,  3.33333333e-01,  3.33333333e-01,  3.33333333e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_basis[] = {
  3.75000000e-01, -9.37500000e-02,
 -9.37500000e-02,  6.25000000e-02,
  3.75000000e-01,  3.75000000e-01,
  0.00000000e+00,  6.56250000e-01,
 -9.37500000e-02,  4.37500000e-01,
 -0.00000000e+00, -0.00000000e+00,
  0.00000000e+00, -9.37500000e-02,
  6.56250000e-01,  4.37500000e-01,
 -0.00000000e+00, -0.00000000e+00,
 -9.37500000e-02,  0.00000000e+00,
 -9.37500000e-02,  2.50000000e-01,
  1.87500000e-01,  7.50000000e-01,
 -9.37500000e-02, -9.37500000e-02,
  0.00000000e+00,  2.50000000e-01,
  7.50000000e-01,  1.87500000e-01,
  3.75000000e-01,  1.56250000e-01,
  1.56250000e-01,  1.56250000e+00,
 -6.25000000e-01, -6.25000000e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_basisDerivRef[] = {
 -1.00000000e+00, -1.00000000e+00,
 -2.50000000e-01,  0.00000000e+00,
  0.00000000e+00, -2.50000000e-01,
  2.50000000e-01,  2.50000000e-01,
 -2.50000000e-01,  1.25000000e+00,
  1.25000000e+00, -2.50000000e-01,
  5.00000000e-01,  5.00000000e-01,
  1.25000000e+00,  0.00000000e+00,
  0.00000000e+00, -2.50000000e-01,
  2.50000000e-01,  1.75000000e+00,
 -2.50000000e-01, -2.50000000e-01,
 -1.75000000e+00, -1.75000000e+00,
  5.00000000e-01,  5.00000000e-01,
 -2.50000000e-01,  0.00000000e+00,
  0.00000000e+00,  1.25000000e+00,
  1.75000000e+00,  2.50000000e-01,
 -1.75000000e+00, -1.75000000e+00,
 -2.50000000e-01, -2.50000000e-01,
 -2.50000000e-01, -2.50000000e-01,
  5.00000000e-01,  0.00000000e+00,
  0.00000000e+00, -2.50000000e-01,
  2.50000000e-01,  1.00000000e+00,
 -2.50000000e-01,  5.00000000e-01,
 -2.50000000e-01, -1.00000000e+00,
 -2.50000000e-01, -2.50000000e-01,
 -2.50000000e-01,  0.00000000e+00,
  0.00000000e+00,  5.00000000e-01,
  1.00000000e+00,  2.50000000e-01,
 -1.00000000e+00, -2.50000000e-01,
  5.00000000e-01, -2.50000000e-01,
  1.00000000e+00,  1.00000000e+00,
  7.50000000e-01,  0.00000000e+00,
  0.00000000e+00,  7.50000000e-01,
  1.25000000e+00,  1.25000000e+00,
 -1.25000000e+00, -1.75000000e+00,
 -1.75000000e+00, -1.25000000e+00,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_fieldTIncr[] = {
 -4.00000000e-01, -6.00000000e-01,
  7.00000000e-01,  8.00000000e-01,
  0.00000000e+00,  2.00000000e-01,
 -5.00000000e-01, -4.00000000e-01,
  3.00000000e-01,  9.00000000e-01,
 -3.00000000e-01, -9.00000000e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_fieldT[] = {
 -3.00000000e-01, -4.00000000e-01,
  5.00000000e-01,  6.00000000e-01,
  0.00000000e+00,  1.00000000e-01,
 -2.00000000e-01, -3.00000000e-01,
  2.00000000e-01,  3.00000000e-01,
 -1.00000000e-01, -2.00000000e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_fieldTmdt[] = {
 -2.00000000e-01, -3.00000000e-01,
  3.00000000e-01,  4.00000000e-01,
  0.00000000e+00, -1.00000000e-01,
 -3.00000000e-01, -2.00000000e-01,
  1.00000000e-01,  4.00000000e-01,
 -2.00000000e-01, -6.00000000e-01,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_valsResidual[] = {
  1.29278791e+09,  2.30117470e+11,
 -1.01094274e+11, -3.41937391e+10,
 -6.14324363e+09,  2.06921658e+11,
  2.08592314e+11,  1.56195193e+11,
 -1.44154554e+11, -5.45281912e+11,
  4.15069698e+10, -1.37586697e+10,
};

const double pylith::feassemble::ElasticityImplicitData2DQuadratic::_valsJacobian[] = {
  4.84911024e+10,  1.08235677e+10,
  1.04859303e+10,  1.58599175e+10,
  1.60687211e+10, -9.93272569e+09,
  3.11728516e+10,  6.95800781e+09,
 -5.77211733e+10,  9.45258247e+09,
 -4.84974320e+10, -3.31613498e+10,
  1.08235677e+10,  1.37388672e+11,
  1.49614800e+10,  1.68082682e+10,
 -9.03428819e+09,  5.84283854e+10,
  6.95800781e+09,  8.83212891e+10,
  7.96820747e+09, -1.84855143e+11,
 -3.16769748e+10, -1.16091471e+11,
  1.04859303e+10,  1.49614800e+10,
  4.98634621e+10,  1.05658637e+10,
  3.04108796e+09, -1.57118056e+09,
 -2.96531395e+10,  2.41807726e+10,
  2.28334780e+09, -1.00401476e+10,
 -3.60206887e+10, -3.80967882e+10,
  1.58599175e+10,  1.68082682e+10,
  1.05658637e+10,  2.16878255e+10,
 -1.72743056e+09, -8.07291667e+08,
  2.63292101e+10,  1.63899740e+10,
 -1.07823351e+10, -9.12434896e+09,
 -4.02452257e+10, -4.49544271e+10,
  1.60687211e+10, -9.03428819e+09,
  3.04108796e+09, -1.72743056e+09,
  5.98153935e+10, -3.38107639e+10,
 -2.72258391e+10,  1.55056424e+10,
 -5.15554109e+10,  2.90256076e+10,
 -1.43952546e+08,  4.12326389e+07,
 -9.93272569e+09,  5.84283854e+10,
 -1.57118056e+09, -8.07291667e+08,
 -3.38107639e+10,  9.58802083e+10,
  1.33572049e+10,  3.44856771e+10,
  3.11740451e+10, -1.60766927e+11,
  7.83420139e+08, -2.72200521e+10,
  3.11728516e+10,  6.95800781e+09,
 -2.96531395e+10,  2.63292101e+10,
 -2.72258391e+10,  1.33572049e+10,
  1.75797635e+11, -1.31429036e+10,
 -6.82493128e+10, -4.63926866e+10,
 -8.18421947e+10,  1.28911675e+10,
  6.95800781e+09,  8.83212891e+10,
  2.41807726e+10,  1.63899740e+10,
  1.55056424e+10,  3.44856771e+10,
 -1.31429036e+10,  2.86053711e+11,
 -4.77208116e+10, -1.61957357e+11,
  1.42192925e+10, -2.63293294e+11,
 -5.77211733e+10,  7.96820747e+09,
  2.28334780e+09, -1.07823351e+10,
 -5.15554109e+10,  3.11740451e+10,
 -6.82493128e+10, -4.77208116e+10,
  1.76619973e+11, -1.47922092e+10,
 -1.37742332e+09,  3.41531033e+10,
  9.45258247e+09, -1.84855143e+11,
 -1.00401476e+10, -9.12434896e+09,
  2.90256076e+10, -1.60766927e+11,
 -4.63926866e+10, -1.61957357e+11,
 -1.47922092e+10,  3.83965169e+11,
  3.27468533e+10,  1.32738607e+11,
 -4.84974320e+10, -3.16769748e+10,
 -3.60206887e+10, -4.02452257e+10,
 -1.43952546e+08,  7.83420139e+08,
 -8.18421947e+10,  1.42192925e+10,
 -1.37742332e+09,  3.27468533e+10,
  1.67881691e+11,  2.41726345e+10,
 -3.31613498e+10, -1.16091471e+11,
 -3.80967882e+10, -4.49544271e+10,
  4.12326389e+07, -2.72200521e+10,
  1.28911675e+10, -2.63293294e+11,
  3.41531033e+10,  1.32738607e+11,
  2.41726345e+10,  3.18820638e+11,
};

pylith::feassemble::ElasticityImplicitData2DQuadratic::ElasticityImplicitData2DQuadratic(void)
{ // constructor
  spaceDim = _spaceDim;
  cellDim = _cellDim;
  numVertices = _numVertices;
  numCells = _numCells;
  numBasis = _numBasis;
  numQuadPts = _numQuadPts;
  matType = const_cast<char*>(_matType);
  matDBFilename = const_cast<char*>(_matDBFilename);
  matId = _matId;
  matLabel = const_cast<char*>(_matLabel);
  dt = _dt;
  gravityVec = const_cast<double*>(_gravityVec);
  vertices = const_cast<double*>(_vertices);
  cells = const_cast<int*>(_cells);
  verticesRef = const_cast<double*>(_verticesRef);
  quadPts = const_cast<double*>(_quadPts);
  quadWts = const_cast<double*>(_quadWts);
  basis = const_cast<double*>(_basis);
  basisDerivRef = const_cast<double*>(_basisDerivRef);
  fieldTIncr = const_cast<double*>(_fieldTIncr);
  fieldT = const_cast<double*>(_fieldT);
  fieldTmdt = const_cast<double*>(_fieldTmdt);
  valsResidual = const_cast<double*>(_valsResidual);
  valsJacobian = const_cast<double*>(_valsJacobian);
} // constructor

pylith::feassemble::ElasticityImplicitData2DQuadratic::~ElasticityImplicitData2DQuadratic(void)
{}


// End of file
