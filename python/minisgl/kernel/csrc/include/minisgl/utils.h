#pragma once

#include <dlpack/dlpack.h>

#include <concepts>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <utility>

namespace minisgl_compat {

#if defined(__has_include)
#if __has_include(<source_location>)
#include <source_location>
using source_location = std::source_location;
#else
struct source_location {
  static constexpr auto current(const char *file = __builtin_FILE(),
                                const char *function = __builtin_FUNCTION(),
                                std::uint_least32_t line = __builtin_LINE(),
                                std::uint_least32_t column = 0) noexcept
      -> source_location {
    return source_location(file, function, line, column);
  }

  constexpr source_location(const char *file = "unknown",
                            const char *function = "",
                            std::uint_least32_t line = 0,
                            std::uint_least32_t column = 0) noexcept
      : m_file(file), m_function(function), m_line(line), m_column(column) {}

  constexpr auto file_name() const noexcept -> const char * { return m_file; }
  constexpr auto function_name() const noexcept -> const char * {
    return m_function;
  }
  constexpr auto line() const noexcept -> std::uint_least32_t { return m_line; }
  constexpr auto column() const noexcept -> std::uint_least32_t {
    return m_column;
  }

private:
  const char *m_file;
  const char *m_function;
  std::uint_least32_t m_line;
  std::uint_least32_t m_column;
};
#endif
#else
#include <source_location>
using source_location = std::source_location;
#endif

} // namespace minisgl_compat

namespace host {

using SourceLocation = minisgl_compat::source_location;

struct PanicError : public std::runtime_error {
public:
  // copy and move constructors
  PanicError(std::string msg) : runtime_error(msg), m_message(std::move(msg)) {}
  auto detail() const -> std::string_view {
    const auto sv = std::string_view{m_message};
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]]
inline auto panic(SourceLocation location, Args &&...args) -> void {
  std::ostringstream os;
  os << "Runtime check failed at " << location.file_name() << ":"
     << location.line();
  if constexpr (sizeof...(args) > 0) {
    os << ": ";
    (os << ... << std::forward<Args>(args));
  } else {
    os << " in " << location.function_name();
  }
  throw PanicError(std::move(os).str());
}

template <typename... Args> struct Panic {
  explicit Panic(Args &&...args,
                 SourceLocation location = SourceLocation::current()) {
    [[unlikely]];
    ::host::panic(location, std::forward<Args>(args)...);
  }
  [[noreturn]] ~Panic() { std::terminate(); }
};

template <typename... Args> struct RuntimeCheck {
  template <typename T>
  explicit RuntimeCheck(
      T &&condition, Args &&...args, SourceLocation location = SourceLocation::current()) {
    if (!condition) {
      [[unlikely]];
      ::host::panic(location, std::forward<Args>(args)...);
    }
  }
};

template <typename T, typename... Args>
explicit RuntimeCheck(T &&, Args &&...) -> RuntimeCheck<Args...>;

template <typename... Args> explicit Panic(Args &&...) -> Panic<Args...>;

template <std::integral T, std::integral U>
inline constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

inline auto dtype_bytes(DLDataType dtype) -> std::size_t {
  return static_cast<std::size_t>(dtype.bits / 8);
}

namespace pointer {

template <typename T, std::integral... U>
inline auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
inline auto offset(const T *ptr, U... offset) -> const void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

} // namespace host
