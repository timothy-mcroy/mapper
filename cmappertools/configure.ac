# Required packages: autoconf, autoconf-archive
#
# Generate the "configure" script from this by invoking
#
#     aclocal
#     autoconf
#
AC_PREREQ(2.13)
AC_INIT([cmappertools], [1.0.18])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_PATH_PYTHON([2.7])
LT_PREREQ([2.4.2])
LT_INIT([disable-static])
AC_CANONICAL_SYSTEM
AM_INIT_AUTOMAKE
AC_PROG_CXX
AC_LANG(C++)

AX_BOOST_BASE([1.47], [],
[echo 'Warning: Path to Boost libraries was not found. Try setting the BOOST_ROOT environment variable.'])

AX_BOOST_THREAD

AX_BOOST_CHRONO

extra_compile_args="$BOOST_CPPFLAGS"
extra_link_args="$BOOST_LDFLAGS $BOOST_THREAD_LIB $BOOST_CHRONO_LIB"

AX_APPEND_COMPILE_FLAGS([-fvisibility=hidden -fno-common -fvisibility-inlines-hidden -ffunction-sections -fdata-sections], [extra_compile_args])

AX_APPEND_LINK_FLAGS([-Wl,--gc-sections -Wl,-s], [extra_link_args])

DEBUGFLAGS='-Wall -Weffc++'
DEBUGFLAGS=$DEBUGFLAGS' -Wextra  -Wall -Wcast-align'
DEBUGFLAGS=$DEBUGFLAGS' -Wchar-subscripts  -Wcomment -Wconversion'
DEBUGFLAGS=$DEBUGFLAGS' -Wsign-conversion'
DEBUGFLAGS=$DEBUGFLAGS' -Wdisabled-optimization'
DEBUGFLAGS=$DEBUGFLAGS' -Wfloat-equal  -Wformat  -Wformat=2'
DEBUGFLAGS=$DEBUGFLAGS' -Wformat-nonliteral -Wformat-security'
DEBUGFLAGS=$DEBUGFLAGS' -Wformat-y2k'
DEBUGFLAGS=$DEBUGFLAGS' -Wimport  -Winit-self  -Winline'
DEBUGFLAGS=$DEBUGFLAGS' -Winvalid-pch  '
DEBUGFLAGS=$DEBUGFLAGS' -Wmissing-braces'
DEBUGFLAGS=$DEBUGFLAGS' -Wmissing-field-initializers'
DEBUGFLAGS=$DEBUGFLAGS' -Wmissing-format-attribute'
DEBUGFLAGS=$DEBUGFLAGS' -Wmissing-include-dirs -Wmissing-noreturn'
DEBUGFLAGS=$DEBUGFLAGS' -Wpacked  -Wparentheses  -Wpointer-arith'
DEBUGFLAGS=$DEBUGFLAGS' -Wredundant-decls -Wreturn-type'
DEBUGFLAGS=$DEBUGFLAGS' -Wsequence-point  -Wshadow -Wsign-compare'
DEBUGFLAGS=$DEBUGFLAGS' -Wstack-protector'
DEBUGFLAGS=$DEBUGFLAGS' -Wstrict-aliasing -Wstrict-aliasing=2 -Wswitch'
DEBUGFLAGS=$DEBUGFLAGS' -Wswitch-enum -Wtrigraphs  -Wuninitialized'
DEBUGFLAGS=$DEBUGFLAGS' -Wunknown-pragmas  -Wunreachable-code -Wunused'
DEBUGFLAGS=$DEBUGFLAGS' -Wunused-function  -Wunused-label  -Wunused-parameter'
DEBUGFLAGS=$DEBUGFLAGS' -Wunused-value  -Wunused-variable  -Wvariadic-macros'
DEBUGFLAGS=$DEBUGFLAGS' -Wvolatile-register-var  -Wwrite-strings'
DEBUGFLAGS=$DEBUGFLAGS' -Wlong-long'
DEBUGFLAGS=$DEBUGFLAGS' -Wpadded'
DEBUGFLAGS=$DEBUGFLAGS' -Wcast-qual'
DEBUGFLAGS=$DEBUGFLAGS' -Wswitch-default'
DEBUGFLAGS=$DEBUGFLAGS' -Wnon-virtual-dtor -Wold-style-cast'
DEBUGFLAGS=$DEBUGFLAGS' -Woverloaded-virtual'
DEBUGFLAGS=$DEBUGFLAGS' -Werror'
DEBUGFLAGS=$DEBUGFLAGS' -Waggregate-return'
#DEBUGFLAGS=$DEBUGFLAGS' #-pedantic -pedantic-errors'
#DEBUGFLAGS=$DEBUGFLAGS' -Wl,--print-gc-sections'

#extra_compile_args=${extra_compile_args:+${extra_compile_args} }$DEBUGFLAGS

echo $extra_compile_args > extra_compile_args.txt
echo $extra_link_args > extra_link_args.txt

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
