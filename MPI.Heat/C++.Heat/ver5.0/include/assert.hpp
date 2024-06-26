/**
 * @file assert.hpp
 * @brief This file defines a macro for custom assertions with messages.
 * 
 * This header file provides a macro FINAL_PROJECT_ASSERT_MSG that can 
 * be used for assertions with custom messages. It is based on the 
 * standard assert macro.
 * 
 * @date Jun 26, 2024
 * @version 5.0
 */

#ifndef FINAL_PROJECT_ASSERT_LIYIHAI
#define FINAL_PROJECT_ASSERT_LIYIHAI

#include <cassert>
#include <source_location>
#include <string>


#define FINAL_PROJECT_ASSERT_MSG(expr, msg) assert((expr) && (msg))

// std::string sourceline(const std::source_location location)
// {
//     auto line {location.line()};
//     auto column {location.column()};
//     std::string result {"file: "};
//     result += location.file_name();
//     result += "(" + std::to_string(line) + ":" + std::to_string(column) + ")";
//     result += " '";
//     result += location.function_name();
//     return result;   
// }



#endif