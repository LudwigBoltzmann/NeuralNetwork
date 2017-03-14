#pragma once


#include <exception>
#include <string>
#include <iostream>
#include "Pandora/util/coloredstream.h"

namespace Pandora
{

/// error exception for Pandora
class error : public std::exception
{
private:
    std::string     m_msg;

public:
    explicit error(const std::string& msg) : m_msg(msg) {}
    const char* what(void) const throw() override { return m_msg.c_str(); }
};

class warning {
private:
    std::string m_msg;
    std::string m_msg_header = std::string("[warning] : ");
public:
    explicit warning(const std::string &msg) : m_msg(msg) {
        ColorModifier YELLOW(color::YELLOW);
        ColorModifier DEFAULT(color::DEFAULT);
        std::cout<<YELLOW<<m_msg_header + m_msg<<DEFAULT<<std::endl;
    }
};

class information
{
private:
    std::string m_msg;
    std::string m_msg_header = std::string("[information] : ");
public:
    information(const std::string& msg) : m_msg(msg) {
        ColorModifier LCYAN(color::LCYAN);
        ColorModifier DEFAULT(color::DEFAULT);
        std::cout<<LCYAN<<m_msg_header + m_msg<<DEFAULT<<std::endl;

    }
};


}
