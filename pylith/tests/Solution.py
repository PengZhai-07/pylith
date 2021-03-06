#!/usr/bin/env python
#
# ----------------------------------------------------------------------
#
# Brad T. Aagaard, U.S. Geological Survey
# Charles A. Williams, GNS Science
# Matthew G. Knepley, University of Chicago
#
# This code was developed as part of the Computational Infrastructure
# for Geodynamics (http://geodynamics.org).
#
# Copyright (c) 2010-2017 University of California, Davis
#
# See COPYING for license information.
#
# ----------------------------------------------------------------------
#

## @file pylith/tests/Solution.py
##
## @brief Check displacement solution output from PyLith.

import numpy
import h5py

def check_displacements(testcase, filename, mesh):
  """
  Check displacements.
  """
  h5 = h5py.File(filename, "r", driver="sec2")
  
  # Check vertices
  vertices = h5['geometry/vertices'][:]
  (nvertices, spaceDim) = vertices.shape
  testcase.assertEqual(mesh['nvertices'], nvertices)
  testcase.assertEqual(mesh['spaceDim'], spaceDim)

  # Check displacement solution
  toleranceAbsMask = 0.1
  tolerance = 1.0e-5

  dispE = testcase.calcDisplacements(vertices)
  disp = h5['vertex_fields/displacement'][:]

  (nstepsE, nverticesE, ncompsE) = dispE.shape
  (nsteps, nvertices, ncomps) = disp.shape
  testcase.assertEqual(nstepsE, nsteps)
  testcase.assertEqual(nverticesE, nvertices)
  testcase.assertEqual(ncompsE, ncomps)

  from spatialdata.units.NondimElasticQuasistatic import NondimElasticQuasistatic
  normalizer = NondimElasticQuasistatic()
  normalizer._configure()

  scale = 1.0
  scale *= normalizer.lengthScale().value

  for istep in xrange(nsteps):
    for icomp in xrange(ncomps):
      okay = numpy.zeros((nvertices,), dtype=numpy.bool)

      maskR = numpy.abs(dispE[istep,:,icomp]) > toleranceAbsMask
      ratio = numpy.abs(1.0 - disp[istep,maskR,icomp] / dispE[istep,maskR,icomp])
      if len(ratio) > 0:
        okay[maskR] = ratio < tolerance

      maskD = ~maskR
      diff = numpy.abs(disp[istep,maskD,icomp] - dispE[istep,maskD,icomp]) / scale
      if len(diff) > 0:
        okay[maskD] = diff < tolerance

      if numpy.sum(okay) != nvertices:
        print "Error in component %d of displacement field at time step %d." % (icomp, istep)
        print "Expected values: ",dispE[istep,:,:]
        print "Output values: ",disp[istep,:,:]
        print "Expected values (not okay): ",dispE[istep,~okay,icomp]
        print "Computed values (not okay): ",disp[istep,~okay,icomp]
        print "Relative diff (not okay): ",diff[~okay]
        print "Coordinates (not okay): ",vertices[~okay,:]
        h5.close()
      testcase.assertEqual(nvertices, numpy.sum(okay))    
    
  h5.close()
  return


# End of file
