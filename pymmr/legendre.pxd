
cdef extern from "boost/math/special_functions/legendre.hpp" namespace "boost::math" nogil:
    cdef double legendre_p(int, double)
    cdef double legendre_p_prime(int, double)