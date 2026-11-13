#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#define SR_EXPORT __declspec(dllexport)
#else
#define SR_EXPORT
#endif

// UTF-8 is validated by the host. Literal matching preserves overlaps and NULs.
// Return a Unicode code-point offset, -1 for absence, -2 for invalid input/error.
extern "C" SR_EXPORT std::int64_t sr_find_occurrence(
    const unsigned char* text, std::size_t size,
    const unsigned char* needle, std::size_t width, std::uint64_t occurrence) noexcept {
    if (!text || !needle || width == 0 || size > 32 * 1024 * 1024 || width > 1024 * 1024)
        return -2;
    if (width > size) return -1;
    try {
        std::vector<std::size_t> prefix(width, 0);
        for (std::size_t i = 1, j = 0; i < width; ++i) {
            while (j && needle[i] != needle[j]) j = prefix[j - 1];
            if (needle[i] == needle[j]) ++j;
            prefix[i] = j;
        }
        for (std::size_t i = 0, j = 0; i < size; ++i) {
            while (j && text[i] != needle[j]) j = prefix[j - 1];
            if (text[i] == needle[j]) ++j;
            if (j == width) {
                if (occurrence == 0) {
                    const std::size_t start = i + 1 - width;
                    std::int64_t points = 0;
                    for (std::size_t k = 0; k < start; ++k)
                        if ((text[k] & 0xc0) != 0x80) ++points;
                    return points;
                }
                --occurrence;
                j = prefix[j - 1];
            }
        }
        return -1;
    } catch (...) { return -2; }
}
