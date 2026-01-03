
#ifndef OPENGJK_EXPORT_H
#define OPENGJK_EXPORT_H

#ifdef OPENGJK_STATIC_DEFINE
#  define OPENGJK_EXPORT
#  define OPENGJK_NO_EXPORT
#else
#  ifndef OPENGJK_EXPORT
#    ifdef opengjk_scalar_static_EXPORTS
        /* We are building this library */
#      define OPENGJK_EXPORT 
#    else
        /* We are using this library */
#      define OPENGJK_EXPORT 
#    endif
#  endif

#  ifndef OPENGJK_NO_EXPORT
#    define OPENGJK_NO_EXPORT 
#  endif
#endif

#ifndef OPENGJK_DEPRECATED
#  define OPENGJK_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef OPENGJK_DEPRECATED_EXPORT
#  define OPENGJK_DEPRECATED_EXPORT OPENGJK_EXPORT OPENGJK_DEPRECATED
#endif

#ifndef OPENGJK_DEPRECATED_NO_EXPORT
#  define OPENGJK_DEPRECATED_NO_EXPORT OPENGJK_NO_EXPORT OPENGJK_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef OPENGJK_NO_DEPRECATED
#    define OPENGJK_NO_DEPRECATED
#  endif
#endif

#endif /* OPENGJK_EXPORT_H */
