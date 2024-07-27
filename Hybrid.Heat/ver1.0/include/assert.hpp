///
/// @file assert.hpp
/// @brief Defines the assert using `assert`
/// @author LI Yihai
/// @version 6.0
/// 
#ifndef FINAL_PROJECT_ASSERT_HPP_LIYIHAI
#define FINAL_PROJECT_ASSERT_HPP_LIYIHAI


///
/// FINAL_PROJECT_ASSERT, FINAL_PROJECT_ASSERT_MSG, FINAL_PROJECT_ASSERT_IS_VOID
/// 
#undef FINAL_PROJECT_ASSERT
#undef FINAL_PROJECT_ASSERT_MSG
#undef FINAL_PROJECT_ASSERT_IS_VOID

#if defined(FINAL_PROJECT_DISABLE_ASSERTS) || defined(NDEBUG)

#define FINAL_PROJECT_ASSERT(expr) ((void)0)
#define FINAL_PROJECT_ASSERT_MSG(expr, msg) ((void)0)
#define FINAL_PROJECT_ASSERT_IS_VOID

#else

#pragma once
#include <cassert>
#include <cstdlib>
#include <iostream>

#define FINAL_PROJECT_ASSERT(expr) \
  if (!(expr)) { \
    std::cerr << "Assertion Failed: " #expr \
              << ", file " << __FILE__ \
              << ", line " << __LINE__ \
              << std::endl; \
    assert(expr); \
  }

#define FINAL_PROJECT_ASSERT_MSG(expr, msg) \
  if (!(expr)) { \
    std::cerr << "Assertion Failed: " #expr \
              << ", message " << msg \
              << ", file " << __FILE__ \
              << ", line " << __LINE__ \
              << std::endl; \
    assert ((expr) && (msg)); \
  }

#if defined(NDEBUG)
#define FINAL_PROJECT_ASSERT_IS_VOID
#endif

#endif // end of define FINAL_PROJECT_ASSERT, FINAL_PROJECT_ASSERT_MSG, FINAL_PROJECT_ASSERT_IS_VOID




#endif // end of define FINAL_PROJECT_ASSERT_HPP_LIYIHAI