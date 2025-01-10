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

#include <assert.h>

#undef __FUNCT__
#if defined(__FUNCTION_NAME__)
#define __FUNCT__ __FUNCTION_NAME__
#undef PETSC_FUNCTION_NAME
#define PETSC_FUNCTION_NAME __FUNCT__
#else
#define __FUNCT__ __func__
#endif

#define PYLITH_METHOD_BEGIN PetscFunctionBeginUser
#define PYLITH_METHOD_END PetscFunctionReturnVoid()
#define PYLITH_METHOD_RETURN(v) PetscFunctionReturn(v)

#define PYLITH_CHECK_ERROR(err) do {if (PetscUnlikely(err)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,err,PETSC_ERROR_REPEAT,0);throw std::runtime_error("Error detected while in PETSc function.");}} while (0)
#define PYLITH_CHECK_ERROR_NOTHROW(err) do {if (PetscUnlikely(err)) {PetscError(PETSC_COMM_SELF,__LINE__,PETSC_FUNCTION_NAME,__FILE__,err,PETSC_ERROR_REPEAT,0);}} while (0)

#define PYLITH_ERROR_RETURN(comm,error,msg) SETERRQ(comm,error,"%s",msg)

// End of file
