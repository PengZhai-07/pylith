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
// This file was generated from python application elasticitylgdeformapp.

#if !defined(pylith_feassemble_elasticityimplicitlgdeformgravdata2dquadratic_hh)
#define pylith_feassemble_elasticityimplicitlgdeformgravdata2dquadratic_hh

#include "IntegratorData.hh"

namespace pylith {
  namespace feassemble {
     class ElasticityImplicitLgDeformGravData2DQuadratic;
  } // pylith
} // feassemble

class pylith::feassemble::ElasticityImplicitLgDeformGravData2DQuadratic : public IntegratorData
{

public: 

  /// Constructor
  ElasticityImplicitLgDeformGravData2DQuadratic(void);

  /// Destructor
  ~ElasticityImplicitLgDeformGravData2DQuadratic(void);

private:

  static const int _spaceDim;

  static const int _cellDim;

  static const int _numVertices;

  static const int _numCells;

  static const int _numBasis;

  static const int _numQuadPts;

  static const char* _matType;

  static const char* _matDBFilename;

  static const int _matId;

  static const char* _matLabel;

  static const double _dt;

  static const double _gravityVec[];

  static const double _vertices[];

  static const int _cells[];

  static const double _verticesRef[];

  static const double _quadPts[];

  static const double _quadWts[];

  static const double _basis[];

  static const double _basisDerivRef[];

  static const double _fieldTIncr[];

  static const double _fieldT[];

  static const double _fieldTmdt[];

  static const double _valsResidual[];

  static const double _valsJacobian[];

};

#endif // pylith_feassemble_elasticityimplicitlgdeformgravdata2dquadratic_hh

// End of file
