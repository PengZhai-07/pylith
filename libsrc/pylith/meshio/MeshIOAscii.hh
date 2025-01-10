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

#include "pylith/meshio/MeshIO.hh" // ISA MeshIO

#include "spatialdata/utils/utilsfwd.hh" // USES LineParser

#include <iosfwd> // USES std::istream, std::ostream
#include <string> // HASA std::string

class pylith::meshio::MeshIOAscii : public MeshIO {
    friend class TestMeshIOAscii; // unit testing

    // PUBLIC METHODS //////////////////////////////////////////////////////////////////////////////////////////////////
public:

    /// Constructor
    MeshIOAscii(void);

    /// Destructor
    ~MeshIOAscii(void);

    /// Deallocate PETSc and local data structures.
    void deallocate(void);

    /** Set filename for ASCII file.
     *
     * @param filename Name of file
     */
    void setFilename(const char* name);

    /** Get filename of ASCII file.
     *
     * @returns Name of file
     */
    const char* getFilename(void) const;

    // PROTECTED METHODS ///////////////////////////////////////////////////////////////////////////////////////////////
protected:

    /// Write mesh
    void _write(void) const;

    /// Read mesh
    void _read(void);

    // PRIVATE METHODS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    /** Read mesh vertices.
     *
     * @param parser Input parser.
     * @param coordinates Pointer to array of vertex coordinates
     * @param numVertices Pointer to number of vertices
     * @param spaceDim Pointer to dimension of coordinates vector space
     */
    void _readVertices(spatialdata::utils::LineParser& parser,
                       scalar_array* coordinates,
                       int* numVertices,
                       int* spaceDim) const;

    /** Write mesh vertices.
     *
     * @param fileout Output stream
     */
    void _writeVertices(std::ostream& fileout) const;

    /** Read mesh cells.
     *
     * @param parser Input parser.
     * @param pCells Pointer to array of indices of cell vertices
     * @param pMaterialIds Pointer to array of material identifiers
     * @param pNumCells Pointer to number of cells
     * @param pNumCorners Pointer to number of corners
     */
    void _readCells(spatialdata::utils::LineParser& parser,
                    int_array* pCells,
                    int_array* pMaterialIds,
                    int* numCells,
                    int* numCorners) const;

    /** Write mesh cells.
     *
     * @param fileout Output stream
     * @param cells Array of indices of cell vertices
     * @param numCells Number of cells
     * @param numCorners Number of corners
     */
    void _writeCells(std::ostream& fileout) const;

    /** Read a point group.
     *
     * @param parser Input parser.
     * @param mesh The mesh
     */
    void _readGroup(spatialdata::utils::LineParser& parser,
                    int_array* points,
                    pylith::meshio::MeshBuilder::GroupPtType* type,
                    std::string* name) const;

    /** Write a point group.
     *
     * @param fileout Output stream
     * @param name The group name
     */
    void _writeGroup(std::ostream& fileout,
                     const char* name) const;

    // PRIVATE MEMBERS /////////////////////////////////////////////////////////////////////////////////////////////////
private:

    std::string _filename; ///< Name of file
    bool _useIndexZero; ///< Flag indicating if indicates start at 0 (T) or 1 (F)

}; // MeshIOAscii

#include "MeshIOAscii.icc" // inline methods

// End of file
