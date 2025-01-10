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

#include "pylith/problems/ObserversSoln.hh" // Implementation of class methods

#include "pylith/problems/ObserverSoln.hh" // USES ObserverSoln
#include "pylith/topology/Field.hh" // USES Field

#include "pylith/utils/error.hh" // USES PYLITH_METHOD_BEGIN/END
#include "pylith/utils/journals.hh" // USES PYLITH_JOURNAL_DEBUG

#include <typeinfo> // USES typeid()

// ----------------------------------------------------------------------
// Constructor.
pylith::problems::ObserversSoln::ObserversSoln(void) {
    GenericComponent::setName("observerssoln");
} // constructor


// ----------------------------------------------------------------------
// Destructor
pylith::problems::ObserversSoln::~ObserversSoln(void) {
    deallocate();
} // destructor


// ----------------------------------------------------------------------
// Deallocate PETSc and local data structures.
void
pylith::problems::ObserversSoln::deallocate(void) {
    _observers.clear(); // Memory allocation of Observer* managed elsewhere.
} // deallocate


// ----------------------------------------------------------------------
// Register observer to receive notifications.
void
pylith::problems::ObserversSoln::registerObserver(pylith::problems::ObserverSoln* observer) {
    PYLITH_METHOD_BEGIN;
    PYLITH_JOURNAL_DEBUG("registerObserver(observer="<<typeid(observer).name()<<")");

    if (observer) {
        observer->index = _observers.size();
        _observers.insert(observer);
    } // if

    PYLITH_METHOD_END;
} // registerObserver


// ----------------------------------------------------------------------
// Remove observer from receiving notifications.
void
pylith::problems::ObserversSoln::removeObserver(pylith::problems::ObserverSoln* observer) {
    PYLITH_METHOD_BEGIN;
    PYLITH_JOURNAL_DEBUG("removeObserver(observer="<<typeid(observer).name()<<")");

    if (observer) {
        _observers.erase(observer);
    } // if

    PYLITH_METHOD_END;
} // removeObserver


// ----------------------------------------------------------------------
// Set time scale in observers.
void
pylith::problems::ObserversSoln::setTimeScale(const PylithReal value) {
    PYLITH_METHOD_BEGIN;
    PYLITH_JOURNAL_DEBUG("setTimeScale(value="<<value<<")");

    for (iterator iter = _observers.begin(); iter != _observers.end(); ++iter) {
        assert(*iter);
        (*iter)->setTimeScale(value);
    } // for

    PYLITH_METHOD_END;
} // setTimeScale


// ----------------------------------------------------------------------
// Verify observers.
void
pylith::problems::ObserversSoln::verifyObservers(const pylith::topology::Field& solution) const {
    PYLITH_METHOD_BEGIN;
    PYLITH_JOURNAL_DEBUG("verifyObservers(solution="<<solution.getLabel()<<")");

    for (iterator iter = _observers.begin(); iter != _observers.end(); ++iter) {
        assert(*iter);
        (*iter)->verifyConfiguration(solution);
    } // for

    PYLITH_METHOD_END;
} // verifyObservers


// ------------------------------------------------------------------------------------------------
// Notify observers.
void
pylith::problems::ObserversSoln::notifyObservers(const PylithReal t,
                                                 const PylithInt tindex,
                                                 const pylith::topology::Field& solution,
                                                 const pylith::problems::Observer::NotificationType notification) {
    PYLITH_METHOD_BEGIN;
    PYLITH_JOURNAL_DEBUG("notifyObservers(t="<<t<<", tindex="<<tindex<<", solution="<<solution.getLabel()<<")");

    for (iterator iter = _observers.begin(); iter != _observers.end(); ++iter) {
        assert(*iter);
        (*iter)->update(t, tindex, solution, notification);
    } // for

    PYLITH_METHOD_END;
} // notifyObservers


// ------------------------------------------------------------------------------------------------
// Comparison function for keeping set of observers in order.
bool
pylith::problems::ObserversSoln::_compare::operator()(const ObserverSoln* const a,
                                                      const ObserverSoln* const b) const {
    assert(a);
    assert(b);
    return a->index < b->index;
} // _compare


// End of file
