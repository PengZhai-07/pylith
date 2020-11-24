import numpy

numpy.set_printoptions(linewidth=400)

LAGRANGE_PC = "jacobi"

K = numpy.array([
    [1.68056, -0.395833, 0.652778, 0.0208333, 0.0, 0.0, 0.0, 0.0],
    [-0.395833, 0.888889, -0.0208333, -0.138889, 0.0, 0.0, 0.0, 0.0],
    [0.652778, -0.0208333, 1.68056, 0.395833, 0.0, 0.0, 0.0, 0.0],
    [0.0208333, -0.138889, 0.395833, 0.888889, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.68056, 0.395833, 0.652778, -0.0208333],
    [0.0, 0.0, 0.0, 0.0, 0.395833, 0.888889, 0.0208333, -0.138889],
    [0.0, 0.0, 0.0, 0.0, 0.652778, 0.0208333, 1.68056, -0.395833],
    [0.0, 0.0, 0.0, 0.0, -0.0208333, -0.138889, -0.395833, 0.888889],
    ], dtype=numpy.float64)

L3 = numpy.array([
    [-0.666667, 0, -0.333333, 0, 0.666667, 0, 0.333333, 0],
    [0, -0.666667, 0, -0.333333, 0, 0.666667, 0, 0.333333],
    [-0.333333, 0, -0.666667, 0, 0.333333, 0, 0.666667, 0],
    [0, -0.333333, 0, -0.666667, 0, 0.333333, 0, 0.666667],
    ], dtype=numpy.float64)

L2 = numpy.array([
    [-1, 0, 0, 0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0, 0, 1, 0],
    [0, 0, 0, -1, 0, 0, 0, 1],
    ], dtype=numpy.float64)

KDiagInv = numpy.diagflat(1.0/numpy.diag(K))
Kinv = numpy.linalg.inv(K)

zero4 = numpy.zeros((4,4))

Sp2 = zero4 - numpy.dot(numpy.dot(L2, KDiagInv), L2.T)
Sp3 = zero4 - numpy.dot(numpy.dot(L3, KDiagInv), L3.T)

J2 = numpy.block([[K, L2.T], [L2, zero4]])
J3 = numpy.block([[K, L3.T], [L3, zero4]])


zero48 = numpy.zeros((4,8))

if LAGRANGE_PC == "lu": 
    Sp2inv = numpy.linalg.inv(Sp2)
    Sp3inv = numpy.linalg.inv(Sp3)
elif LAGRANGE_PC == "jacobi":
    Sp2inv = numpy.diagflat(1.0/numpy.diag(Sp2))
    Sp3inv = numpy.diagflat(1.0/numpy.diag(Sp3))
else:
    raise ValueError("Unknown LAGRANGE PC '{}'.".format(LAGRANGE_PC))

P2 = numpy.block([[Kinv, zero48.T], [zero48, -Sp2inv]])
P3 = numpy.block([[Kinv, zero48.T], [zero48, -Sp3inv]])

R2 = numpy.dot(P2, J2)
R3 = numpy.dot(P3, J3)

R3_lu = numpy.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.429356, -0.328738, -0.0326623, -0.266303],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.192826, -0.990245, 0.00552111, -0.640609],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0326623, 0.266303, -0.429356, 0.328738],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.00552111, -0.640609, 0.192826, -0.990245],

    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.429356, -0.328738, 0.0326623, -0.266303],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.192826, 0.990245, 0.00552111, 0.640609],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0326623, 0.266303, 0.429356, 0.328738],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.00552111, 0.640609, 0.192826, 0.990245],
    
    [-1.68056, 0.0, 0.840278, 0.0, 1.68056, 0.0, -0.840278, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.888889, 0.0, 0.444444, 0.0, 0.888889, 0.0, -0.444444, 0.0, 0.0, 0.0, 0.0],
    [0.840278, 0.0, -1.68056, 0.0, -0.840278, 0.0, 1.68056, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.444444, 0.0, -0.888889, 0.0, -0.444444, 0.0, 0.888889, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

R3_jacobi = numpy.array([ # with displacement LU and Lagrange multiplier rowsum; Schur preconditioner scaled by +1.0 not -1.0
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.429356, -0.328738, -0.0326623, -0.266303],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.192826, -0.990245, 0.00552111, -0.640609],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0326623, 0.266303, -0.429356, 0.328738],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.00552111, -0.640609, 0.192826, -0.990245],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.429356, -0.328738, 0.0326623, -0.266303],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -0.192826, 0.990245, 0.00552111, 0.640609],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0326623, 0.266303, 0.429356, 0.328738],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.00552111, 0.640609, 0.192826, 0.990245],
    [0.560185, 0.0, 0.280093, 0.0, -0.560185, 0.0, -0.280093, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.296296, 0.0, 0.148148, 0.0, -0.296296, 0.0, -0.148148, 0.0, 0.0, 0.0, 0.0],
    [0.280093, 0.0, 0.560185, 0.0, -0.280093, 0.0, -0.560185, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.148148, 0.0, 0.296296, 0.0, -0.148148, 0.0, -0.296296, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

R3_upperuser = numpy.array([
    [1.5, 0.135859, 0.0, 0.0465734, -0.5, -0.135859, 0.0, -0.0465734, -0.429356, -0.328738, -0.0326623, -0.266303],
    [0.226351, 1.5, -0.0236487, 0.0, -0.226351, -0.5, 0.0236487, 0.0, -0.192826, -0.990245, 0.00552111, -0.640609],
    [0.0, -0.0465734, 1.5, -0.135859, 0.0, 0.0465734, -0.5, 0.135859, -0.0326623, 0.266303, -0.429356, 0.328738],
    [0.0236487, 0.0, -0.226351, 1.5, -0.0236487, 0.0, 0.226351, -0.5, -0.00552111, -0.640609, 0.192826, -0.990245],
    [-0.5, 0.135859, 0.0, 0.0465734, 1.5, -0.135859, 0.0, -0.0465734, 0.429356, -0.328738, 0.0326623, -0.266303],
    [0.226351, -0.5, -0.0236487, 0.0, -0.226351, 1.5, 0.0236487, 0.0, -0.192826, 0.990245, 0.00552111, 0.640609],
    [0.0, -0.0465734, -0.5, -0.13585, 0.0, 0.0465734, 1.5, 0.135859, 0.0326623, 0.266303, 0.429356, 0.328738],
    [0.0236487, 0.0, -0.226351, -0.5, -0.0236487, 0.0, 0.226351, 1.5, -0.00552111, 0.640609, 0.192826, 0.990245],
    [1.17131, 0.0, -0.0891047, 0.0, -1.17131, 0.0, 0.0891047, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.868324, 0.0, -0.561736, 0.0, -0.868324, 0.0, 0.561736, 0.0, 0.0, 0.0, 0.0],
    [-0.0891047, 0.0, 1.17131, 0.0, 0.0891047, 0.0, -1.17131, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.561736, 0.0, 0.868324, 0.0, 0.561736, 0.0, -0.868324, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

R2_upperuser = numpy.array([
    [1.5, 1.3585907326431992e-01, 0.0, 4.6573360322827051e-02, -4.9999999999999978e-01, -1.3585907326431990e-01, 0.0, -4.6573360322827072e-02, -8.2605069351798832e-01, -3.9117310777768105e-01, 3.6403196032364377e-01, -2.0386821914313483e-01],
    [2.2635134961927930e-01, 1.5, -2.3648650380720743e-02, 0.0, -2.2635134961927930e-01, -4.9999999999999978e-01, 2.3648650380720605e-02, 0.0, -3.9117310777768100e-01, -1.3398806944693984e+00, 2.0386821914313480e-01, -2.9097330232434521e-01],
    [0.0, -4.6573360322827023e-02, 1.5, -1.3585907326431995e-01, 0.0, 4.6573360322827030e-02, -5.0000000000000000e-01, 1.3585907326431998e-01, 3.6403196032364377e-01, 2.0386821914313480e-01, -8.2605069351798832e-01, 3.9117310777768116e-01],
    [2.3648650380720591e-02, 0.0, -2.2635134961927966e-01, 1.5, -2.3648650380720549e-02, 0.0, 2.2635134961927939e-01, -4.9999999999999994e-01, -2.0386821914313480e-01, -2.9097330232434515e-01, 3.9117310777768116e-01, -1.3398806944693984e+00],
    [-4.9999999999999978e-01, 1.3585907326431990e-01, 0.0, 4.6573360322827086e-02, 1.5, -1.3585907326431990e-01, 0.0, -4.6573360322827079e-02, 8.2605069351798821e-01, -3.9117310777768100e-01, -3.6403196032364382e-01, -2.0386821914313483e-01],
    [2.2635134961927922e-01, -4.9999999999999983e-01, -2.3648650380720605e-02, 0.0, -2.2635134961927927e-01, 1.5, 2.3648650380720605e-02, 0.0, -3.9117310777768105e-01, 1.3398806944693982e+00, 2.0386821914313480e-01, 2.9097330232434515e-01],
    [0.0, -4.6573360322827044e-02, -0.5, -1.3585907326431998e-01, 0.0, 4.6573360322827044e-02, 1.5, 1.3585907326431990e-01, -3.6403196032364377e-01, 2.0386821914313477e-01, 8.2605069351798832e-01, 3.9117310777768116e-01],
    [2.3648650380720546e-02, 0.0, -2.2635134961927941e-01, -4.9999999999999983e-01, -2.3648650380720615e-02, 0.0, 2.2635134961927944e-01, 1.5, -2.0386821914313480e-01, 2.9097330232434510e-01, 3.9117310777768111e-01, 1.3398806944693982e+00],
    [7.5117304325336720e-01, 0.0, 3.3103415761716304e-01, 0.0, -7.5117304325336720e-01, 0.0, -3.3103415761716304e-01, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.9163717175999463e-01, 0.0, -8.5049334355175366e-02, 0.0, -3.9163717175999463e-01, 0.0, 8.5049334355175366e-02, 0.0, 0.0, 0.0, 0.0],
    [3.3103415761716298e-01, 0.0, 7.5117304325336731e-01, 0.0, -3.3103415761716298e-01, 0.0, -7.5117304325336731e-01, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -8.5049334355175366e-02, 0.0, 3.9163717175999463e-01, 0.0, 8.5049334355175366e-02, 0.0, -3.9163717175999463e-01, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

# custom pc w/additive
R2_customadd = numpy.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.2605069351798821e-01, -3.9117310777768105e-01, 3.6403196032364382e-01, -2.0386821914313485e-01],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9117310777768100e-01, -1.3398806944693982e+00, 2.0386821914313474e-01, -2.9097330232434515e-01],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6403196032364377e-01, 2.0386821914313477e-01, -8.2605069351798832e-01, 3.9117310777768116e-01],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0386821914313480e-01, -2.9097330232434515e-01, 3.9117310777768116e-01, -1.3398806944693984e+00],
    
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.9663786698518925e-05, 3.2427999193310224e-05, -1.0349359124036528e-06, 8.2603140773586026e-01, -3.9114045074977766e-01, -3.6400156917034210e-01, -2.0385119923560852e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 9.9996670288886180e-01, -5.4911025515178014e-05, 1.7524791447454113e-06, -3.9114045074977766e-01, 1.3398253956246635e+00, 2.0381675715238909e-01, 2.9094448215043089e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, -3.0986825013561148e-05, 9.9994889892003258e-01, 1.6308851651936074e-06, -3.6400156917034199e-01, 2.0381675715238920e-01, 8.2600280216502764e-01, 3.9114628726216449e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, -1.7353500573968894e-05, -2.8618053645623753e-05, 1.0000009133419328e+00, -2.0385119923560846e-01, 2.9094448215043089e-01, 3.9114628726216438e-01, 1.3398656742202695e+00],
    
    [8.4027777127240866e-01, 0.0, 0.0, 0.0, -8.4027777127240866e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 4.4444444281810180e-01, 0.0, 0.0, 0.0, -4.4444444281810180e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 8.4027777127240866e-01, 0.0, 0.0, 0.0, -8.4027777127240866e-01, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 4.4444444281810180e-01, 0.0, 0.0, 0.0, -4.4444444281810180e-01, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

# custom pc w/multiplicative
R2_custommult = numpy.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.2605069351798832e-01, -3.9117310777768105e-01, 3.6403196032364377e-01, -2.0386821914313483e-01],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9117310777768100e-01, -1.3398806944693984e+00, 2.0386821914313480e-01, -2.9097330232434521e-01],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6403196032364377e-01, 2.0386821914313480e-01, -8.2605069351798832e-01, 3.9117310777768116e-01],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0386821914313480e-01, -2.9097330232434515e-01, 3.9117310777768116e-01, -1.3398806944693984e+00],
    
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.2605069351798821e-01, -3.9117310777768100e-01, -3.6403196032364382e-01, -2.0386821914313483e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -3.9117310777768105e-01, 1.3398806944693982e+00, 2.0386821914313480e-01, 2.9097330232434515e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.6403196032364377e-01, 2.0386821914313477e-01, 8.2605069351798832e-01, 3.9117310777768116e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0386821914313480e-01, 2.9097330232434510e-01, 3.9117310777768111e-01, 1.3398806944693982e+00],
    
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3882240714146454e+00, 0.0, -6.1177592858535468e-01, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1910050573923661e+00, 0.0, 2.5864293445297332e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.1177592858535457e-01, 0.0, 1.3882240714146454e+00, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5864293445297332e-01, 0.0, 1.1910050573923661e+00],
    ], dtype=numpy.float64)

R2_jacobi = numpy.array([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -8.2605069351798832e-01, -3.9117310777768105e-01, 3.6403196032364377e-01, -2.0386821914313483e-01],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9117310777768100e-01, -1.3398806944693984e+00, 2.0386821914313480e-01, -2.9097330232434521e-01],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6403196032364377e-01, 2.0386821914313480e-01, -8.2605069351798832e-01, 3.9117310777768116e-01],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0386821914313480e-01, -2.9097330232434515e-01, 3.9117310777768116e-01, -1.3398806944693984e+00],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.2605069351798821e-01, -3.9117310777768100e-01, -3.6403196032364382e-01, -2.0386821914313483e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -3.9117310777768105e-01, 1.3398806944693982e+00, 2.0386821914313480e-01, 2.9097330232434515e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -3.6403196032364377e-01, 2.0386821914313477e-01, 8.2605069351798832e-01, 3.9117310777768116e-01],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -2.0386821914313480e-01, 2.9097330232434510e-01, 3.9117310777768111e-01, 1.3398806944693982e+00],
    [-8.4027777127240866e-01, 0.0, 0.0, 0.0, 8.4027777127240866e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -4.4444444281810180e-01, 0.0, 0.0, 0.0, 4.4444444281810180e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -8.4027777127240866e-01, 0.0, 0.0, 0.0, 8.4027777127240866e-01, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -4.4444444281810180e-01, 0.0, 0.0, 0.0, 4.4444444281810180e-01, 0.0, 0.0, 0.0, 0.0],
    ], dtype=numpy.float64)

import pdb; pdb.set_trace()