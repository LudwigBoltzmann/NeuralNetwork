#pragma once


#define PANDORA_SINGLE


namespace Pandora {


#ifdef PANDORA_SINGLE
typedef float float_t;
#else
typedef double float_t;
#endif

}

