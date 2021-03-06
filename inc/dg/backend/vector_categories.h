#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

namespace dg{

struct AnyVectorTag{};

struct StdVectorTag:public AnyVectorTag {};

/**
 * @brief The Thrust Vector concept
 *
 * A Thrust vector must have the two methods begin() and end() which
 * return thrust compatible iterators and the value_type typedef
 */
struct ThrustVectorTag: public AnyVectorTag {};

struct CuspVectorTag: public ThrustVectorTag {};

struct MPIVectorTag: public AnyVectorTag{};


}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
