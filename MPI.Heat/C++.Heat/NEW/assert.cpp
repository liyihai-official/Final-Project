/**
 * @file assert.hpp
 * @brief This file defines a macro for custom assertions with messages.
 * 
 * This header file provides a macro FINAL_PROJECT_ASSERT_MSG that can 
 * be used for assertions with custom messages. It is based on the 
 * standard assert macro.
 * 
 * @date May 25, 2024
 */

#ifndef FINAL_PROJECT_ASSERT_HPP_LIYIHAI
#define FINAL_PROJECT_ASSERT_HPP_LIYIHAI

#include <cassert>

#define FINAL_PROJECT_ASSERT_MSG(expr, msg) assert((expr) && (msg))

#endif