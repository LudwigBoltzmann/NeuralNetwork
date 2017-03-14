#pragma once


#include <ostream>

namespace Pandora
{
enum class color {
    RED       = 31,
    GREEN     = 32,
    YELLOW    = 33,
    BLUE      = 34,
    CYAN      = 36,
    DEFAULT   = 39,
    LRED      = 91,
    LGREEN    = 92,
    LYELLOW   = 93,
    LBLUE     = 94,
    LCYAN     = 96,
    LDEFAULT  = 97
};
class ColorModifier {
    color code;
public:
    ColorModifier(color pCode) : code(pCode) {}
    friend std::ostream&
    operator<<(std::ostream& os, const ColorModifier& mod) {
        int code = static_cast<int>(mod.code);
        return os << "\033[" << code << "m";
    }
};
}
