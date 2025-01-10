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

#include "pylith/utils/GenericComponent.hh" // ISA GenericComponent

#include "pylith/meshio/OutputTriggerStep.hh" // USES OutputTriggerStep

#include "catch2/catch_test_macros.hpp"

// ------------------------------------------------------------------------------------------------
namespace pylith {
    namespace meshio {
        class TestOutputTriggerStep;
    } // meshio
} // pylith

// ------------------------------------------------------------------------------------------------
class pylith::meshio::TestOutputTriggerStep : public pylith::utils::GenericComponent {
    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////
public:

    /// Test setNumStepsSkip() and getNumStepsSkip().
    static
    void testNumStepsSkip(void);

    /// Test shouldWrite().
    static
    void testShouldWrite(void);

}; // TestOutputTriggerStep

// ------------------------------------------------------------------------------------------------
TEST_CASE("TestOutputTriggerStep::testNumStepsSkip", "[TestOutputTriggerStep][testNumStepsSkip]") {
    pylith::meshio::TestOutputTriggerStep::testNumStepsSkip();
}
TEST_CASE("TestOutputTriggerStep::testShouldWrite", "[TestOutputTriggerStep][testShouldWrite]") {
    pylith::meshio::TestOutputTriggerStep::testShouldWrite();
}

// ------------------------------------------------------------------------------------------------
// Test setNumStepsSkip() and getNumStepsSkip().
void
pylith::meshio::TestOutputTriggerStep::testNumStepsSkip(void) {
    OutputTriggerStep trigger;

    int numSkip = 0; // default
    CHECK(numSkip == trigger.getNumStepsSkip());

    numSkip = 2;
    trigger.setNumStepsSkip(numSkip);
    CHECK(numSkip == trigger.getNumStepsSkip());
} // testNumStepsSkip


// ------------------------------------------------------------------------------------------------
// Test shouldWrite().
void
pylith::meshio::TestOutputTriggerStep::testShouldWrite(void) {
    OutputTriggerStep trigger;

    const PylithReal dt = 0.1;
    PylithReal t = 0.0;
    PylithInt tindex = 0;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;

    trigger.setNumStepsSkip(1);
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;

    trigger.setNumStepsSkip(2);
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(true == trigger.shouldWrite(t, tindex++));t += dt;
    CHECK(false == trigger.shouldWrite(t, tindex++));t += dt;
} // testShouldWrite


// End of file
