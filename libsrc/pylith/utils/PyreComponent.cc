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

#include "pylith/utils/PyreComponent.hh" // Implementation of class methods

#include "pylith/utils/journals.hh"

#include <Python.h>

#include <stdexcept> // USES std::logic_error
#include <iostream> // USES std::cerr

// ----------------------------------------------------------------------
// Constructor
pylith::utils::PyreComponent::PyreComponent(void) :
    _name(""),
    _identifier("unknown") { // constructor
    if (!Py_IsInitialized()) {
        throw std::logic_error("Python must be initialized to use PyreComponent in C++.");
    } // if
} // constructor


// ----------------------------------------------------------------------
// Destructor
pylith::utils::PyreComponent::~PyreComponent(void) {}


// ----------------------------------------------------------------------
// Set name of component.
void
pylith::utils::PyreComponent::setName(const char* value) {
    if (!strlen(value)) {
        throw std::logic_error("Cannot set name of Pyre component to empty string.");
    } // if
    _name = value;
    _fullIdentifier = _name + std::string(".") + _identifier;
} // setName


// ----------------------------------------------------------------------
// Get name of component.
const char*
pylith::utils::PyreComponent::getName(void) const {
    return _name.c_str();
} // getName


// ----------------------------------------------------------------------
// Set component identifier (identifies object in component hierarchy).
void
pylith::utils::PyreComponent::setIdentifier(const char* value) {
    if (!strlen(value)) {
        throw std::logic_error("Cannot set name of Pyre identifier to empty string.");
    } // if
    _identifier = value;
    _fullIdentifier = _name + std::string(".") + _identifier;
} // setIdentifier


// ----------------------------------------------------------------------
// Get component identifier (identifies object in component hierarchy).
const char*
pylith::utils::PyreComponent::getIdentifier(void) const {
    return _identifier.c_str();
} // getIdentifier


// ----------------------------------------------------------------------
// Get full name from identifier and name.
const char*
pylith::utils::PyreComponent::getFullIdentifier(void) const {
    return _fullIdentifier.c_str();
} // getFullIdentifier


// End of file
