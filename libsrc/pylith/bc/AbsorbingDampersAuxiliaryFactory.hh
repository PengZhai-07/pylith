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

#include "pylith/bc/bcfwd.hh" // forward declarations
#include "pylith/feassemble/AuxiliaryFactory.hh" // ISA AuxiliaryFactory

class pylith::bc::AbsorbingDampersAuxiliaryFactory : public pylith::feassemble::AuxiliaryFactory {
    friend class TestDirichletAuxiliaryFactory; // unit testing

    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /// Default constructor.
    AbsorbingDampersAuxiliaryFactory(void);

    /// Destructor.
    virtual ~AbsorbingDampersAuxiliaryFactory(void);

    /// Add density field to auxiliary fields.
    void addDensity(void);

    /// Add shear wave speed field to auxiliary fields.
    void addVs(void);

    /// Add dilatational wave speed field to auxiliary fields.
    void addVp(void);

    // NOT IMPLEMENTED /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    AbsorbingDampersAuxiliaryFactory(const AbsorbingDampersAuxiliaryFactory &); ///< Not implemented.
    const AbsorbingDampersAuxiliaryFactory& operator=(const AbsorbingDampersAuxiliaryFactory&); ///< Not implemented

}; // class AbsorbingDampersAuxiliaryFactory

// End of file
