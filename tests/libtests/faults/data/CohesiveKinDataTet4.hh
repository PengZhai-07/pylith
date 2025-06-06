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

#include "CohesiveKinData.hh"

namespace pylith {
    namespace faults {
        class CohesiveKinDataTet4;
    } // pylith
} // faults

class pylith::faults::CohesiveKinDataTet4 : public CohesiveKinData {
    // PUBLIC METHODS ///////////////////////////////////////////////////////
public:

    /// Constructor
    CohesiveKinDataTet4(void);

    /// Destructor
    ~CohesiveKinDataTet4(void);

    // PRIVATE MEMBERS //////////////////////////////////////////////////////
private:

    static const char* _meshFilename; ///< Filename of input mesh

    static const int _spaceDim; ///< Number of dimensions in vertex coordinates
    static const int _cellDim; ///< Number of dimensions associated with cell

    static const int _numBasis; ///< Number of vertices in cell
    static const int _numQuadPts; ///< Number of quadrature points
    static const PylithScalar _quadPts[]; ///< Coordinates of quad pts in ref cell
    static const PylithScalar _quadWts[]; ///< Weights of quadrature points
    static const PylithScalar _basis[]; ///< Basis fns at quadrature points
    static const PylithScalar _basisDeriv[]; ///< Derivatives of basis fns at quad pts
    static const PylithScalar _verticesRef[]; ///< Coordinates of vertices in ref cell (dual basis)

    static const int _id; ///< Fault material identifier
    static const char* _label; ///< Label for fault
    static const char* _finalSlipFilename; ///< Name of db for final slip
    static const char* _slipTimeFilename; ///< Name of db for slip time
    static const char* _riseTimeFilename; ///< Name of db for rise time
    static const char* _matPropsFilename; ///< Name of db for bulk mat properties.
    //@}

    static const PylithScalar _fieldT[]; ///< Field over domain at time t.
    static const PylithScalar _fieldIncr[]; ///< Solution increment field over domain at time t.
    static const PylithScalar _jacobianLumped[]; ///< Lumped Jacobian.

    static const PylithScalar _orientation[]; ///< Expected values for fault orientation.
    static const PylithScalar _area[]; ///< Expected values for fault area.
    static const PylithScalar _residual[]; ///< Expected values from residual calculation.
    static const PylithScalar _jacobian[]; ///< Expected values from Jacobian calculation.
    static const PylithScalar _fieldIncrAdjusted[]; ///< Expected values for colution increment after adjustment.

    static const int _verticesFault[]; ///< Expected points for Fault vertices
    static const int _edgesLagrange[]; ///< Expected points for Lagrange multipliers
    static const int _verticesPositive[]; ///< Expected points for vertices on + side of fault.
    static const int _verticesNegative[]; ///< Expected points for vertices on - side of fault.
    static const int _numFaultVertices; ///< Number of fault vertices

    static const int _numCohesiveCells; ///< Number of cohesive cells
    static const int _cellMappingFault[]; ///< Fault cell
    static const int _cellMappingCohesive[]; ///< Cohesive cell

};

// End of file
