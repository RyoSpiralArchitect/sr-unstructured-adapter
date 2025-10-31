#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum TextType : std::int32_t {
    TYPE_PARAGRAPH = 0,
    TYPE_HEADER = 1,
    TYPE_LIST = 2,
    TYPE_KV = 3,
    TYPE_OTHER = 4,
};

struct TextBlockInput {
    const char *text;
    std::size_t length;
    std::int32_t type_code;
    std::int32_t infer;
    double confidence;
};

struct TextBlockOutput {
    std::size_t offset;
    std::size_t length;
    std::int32_t type_code;
    double confidence;
};

constexpr char32_t BULLET_U2022 = 0x2022;
constexpr char32_t BULLET_U30FB = 0x30FB;

struct CodePointBuffer {
    std::u32string data;
};

bool is_space(char32_t ch) {
    if (ch <= 0x20) {
        return true;
    }
    return ch == 0x3000 || ch == 0x00A0;
}

bool is_digit(char32_t ch) { return ch >= U'0' && ch <= U'9'; }

bool is_alpha(char32_t ch) {
    if (ch >= U'a' && ch <= U'z') {
        return true;
    }
    if (ch >= U'A' && ch <= U'Z') {
        return true;
    }
    return false;
}

bool is_upper(char32_t ch) {
    if (ch >= U'A' && ch <= U'Z') {
        return true;
    }
    return false;
}

char32_t to_upper(char32_t ch) {
    if (ch >= U'a' && ch <= U'z') {
        return static_cast<char32_t>(ch - (U'a' - U'A'));
    }
    return ch;
}

struct DecodeResult {
    std::u32string text;
};

char32_t replacement_char() { return 0xFFFD; }

DecodeResult decode_utf8(const char *text, std::size_t length) {
    DecodeResult result{};
    if (!text || length == 0) {
        return result;
    }
    const unsigned char *data = reinterpret_cast<const unsigned char *>(text);
    std::size_t i = 0;
    while (i < length) {
        unsigned char byte = data[i];
        if (byte < 0x80) {
            result.text.push_back(static_cast<char32_t>(byte));
            ++i;
            continue;
        }
        std::size_t remaining = length - i;
        if ((byte >> 5) == 0x6) {
            if (remaining < 2) {
                result.text.push_back(replacement_char());
                break;
            }
            char32_t cp = static_cast<char32_t>(byte & 0x1F) << 6;
            cp |= static_cast<char32_t>(data[i + 1] & 0x3F);
            result.text.push_back(cp);
            i += 2;
            continue;
        }
        if ((byte >> 4) == 0xE) {
            if (remaining < 3) {
                result.text.push_back(replacement_char());
                break;
            }
            char32_t cp = static_cast<char32_t>(byte & 0x0F) << 12;
            cp |= static_cast<char32_t>(data[i + 1] & 0x3F) << 6;
            cp |= static_cast<char32_t>(data[i + 2] & 0x3F);
            result.text.push_back(cp);
            i += 3;
            continue;
        }
        if ((byte >> 3) == 0x1E) {
            if (remaining < 4) {
                result.text.push_back(replacement_char());
                break;
            }
            char32_t cp = static_cast<char32_t>(byte & 0x07) << 18;
            cp |= static_cast<char32_t>(data[i + 1] & 0x3F) << 12;
            cp |= static_cast<char32_t>(data[i + 2] & 0x3F) << 6;
            cp |= static_cast<char32_t>(data[i + 3] & 0x3F);
            result.text.push_back(cp);
            i += 4;
            continue;
        }
        result.text.push_back(replacement_char());
        ++i;
    }
    return result;
}

std::string encode_utf8(const std::u32string &text) {
    std::string result;
    if (text.empty()) {
        return result;
    }
    result.reserve(text.size() * 2);
    for (char32_t cp : text) {
        if (cp < 0x80) {
            result.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            result.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            result.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            result.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            result.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return result;
}

std::u32string normalize_whitespace(const std::u32string &input) {
    std::u32string step1;
    step1.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        char32_t ch = input[i];
        if (ch == U'\r') {
            if (i + 1 < input.size() && input[i + 1] == U'\n') {
                continue;
            }
            step1.push_back(U'\n');
            continue;
        }
        step1.push_back(ch);
    }

    std::u32string step2;
    step2.reserve(step1.size());
    for (char32_t ch : step1) {
        if (ch == U'\n') {
            while (!step2.empty() && is_space(step2.back())) {
                step2.pop_back();
            }
            step2.push_back(U'\n');
        } else {
            step2.push_back(ch);
        }
    }

    std::u32string step3;
    step3.reserve(step2.size());
    int newline_count = 0;
    for (char32_t ch : step2) {
        if (ch == U'\n') {
            ++newline_count;
            if (newline_count <= 2) {
                step3.push_back(ch);
            }
        } else {
            newline_count = 0;
            step3.push_back(ch);
        }
    }

    if (!step3.empty()) {
        if (step3[0] == BULLET_U2022 || step3[0] == BULLET_U30FB) {
            std::size_t pos = 1;
            while (pos < step3.size() && is_space(step3[pos])) {
                ++pos;
            }
            std::u32string replaced;
            replaced.reserve(step3.size() - pos + 2);
            replaced.push_back(U'-');
            replaced.push_back(U' ');
            replaced.append(step3.begin() + static_cast<std::ptrdiff_t>(pos), step3.end());
            step3.swap(replaced);
        }
    }

    std::size_t start = 0;
    while (start < step3.size() && is_space(step3[start])) {
        ++start;
    }
    std::size_t end = step3.size();
    while (end > start && is_space(step3[end - 1])) {
        --end;
    }
    std::u32string trimmed;
    if (end > start) {
        trimmed.assign(step3.begin() + static_cast<std::ptrdiff_t>(start),
                       step3.begin() + static_cast<std::ptrdiff_t>(end));
    }
    return trimmed;
}

bool all_lines_bullet(const std::u32string &text) {
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find(U'\n', start);
        if (end == std::u32string::npos) {
            end = text.size();
        }
        if (end - start < 2) {
            return false;
        }
        if (text[start] != U'-' || text[start + 1] != U' ') {
            return false;
        }
        start = end + 1;
    }
    return true;
}

bool header_prefix(const std::u32string &text) {
    if (text.empty()) {
        return false;
    }
    std::size_t i = 0;
    if (is_digit(text[0])) {
        while (i < text.size() && is_digit(text[i])) {
            ++i;
        }
        if (i < text.size() && (text[i] == U'.' || text[i] == U')')) {
            ++i;
            while (i < text.size() && is_space(text[i])) {
                ++i;
            }
            return i < text.size();
        }
        return false;
    }
    if (text[0] == U'(' || text[0] == 0xFF08) {
        ++i;
        while (i < text.size() && text[i] != U')' && text[i] != 0xFF09) {
            ++i;
        }
        if (i < text.size()) {
            ++i;
            if (i < text.size() && is_space(text[i])) {
                return true;
            }
        }
    }
    return false;
}

bool is_all_upper(const std::u32string &text) {
    bool seen_alpha = false;
    for (char32_t ch : text) {
        if (is_alpha(ch)) {
            seen_alpha = true;
            if (!is_upper(ch)) {
                return false;
            }
        }
    }
    return seen_alpha;
}

std::int32_t infer_type(const std::u32string &text) {
    if (text.empty()) {
        return TYPE_OTHER;
    }
    int newline_count = 0;
    for (char32_t ch : text) {
        if (ch == U'\n') {
            ++newline_count;
        }
    }
    if (newline_count >= 1 && all_lines_bullet(text)) {
        return TYPE_LIST;
    }
    if (text.size() < 120 && header_prefix(text)) {
        return TYPE_HEADER;
    }
    int words = 0;
    bool in_word = false;
    for (char32_t ch : text) {
        if (is_space(ch)) {
            if (in_word) {
                in_word = false;
            }
        } else {
            if (!in_word) {
                in_word = true;
                ++words;
            }
        }
    }
    if (words <= 6 && is_all_upper(text)) {
        return TYPE_HEADER;
    }
    int colon_count = 0;
    for (char32_t ch : text) {
        if (ch == U':') {
            ++colon_count;
        }
    }
    if (colon_count == 1 && text.size() < 80) {
        return TYPE_KV;
    }
    return TYPE_PARAGRAPH;
}

struct NormalizedResult {
    std::string text;
    std::int32_t type_code;
    double confidence;
};

NormalizedResult process_block(const TextBlockInput &input) {
    DecodeResult decoded = decode_utf8(input.text, input.length);
    std::u32string normalized = normalize_whitespace(decoded.text);
    std::int32_t type_code = input.type_code;
    if (input.infer && input.type_code == TYPE_PARAGRAPH) {
        type_code = infer_type(normalized);
    }
    double confidence = input.confidence;
    if (normalized.empty()) {
        confidence = std::min(confidence, 0.2);
    }
    NormalizedResult result;
    result.text = encode_utf8(normalized);
    result.type_code = type_code;
    result.confidence = confidence;
    return result;
}

} // namespace

extern "C" {

std::size_t normalize_text_blocks(const TextBlockInput *inputs, std::size_t count,
                                  void *buffer, std::size_t buffer_size,
                                  TextBlockOutput *outputs) {
    if (!inputs || !outputs) {
        return 0;
    }
    bool write = buffer != nullptr && buffer_size > 0;
    char *out = static_cast<char *>(buffer);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        NormalizedResult result = process_block(inputs[i]);
        outputs[i].offset = offset;
        outputs[i].length = result.text.size();
        outputs[i].type_code = result.type_code;
        outputs[i].confidence = result.confidence;
        if (write) {
            if (offset + result.text.size() > buffer_size) {
                return offset + result.text.size();
            }
            if (!result.text.empty()) {
                std::memcpy(out + offset, result.text.data(), result.text.size());
            }
        }
        offset += result.text.size();
    }
    return offset;
}

} // extern "C"

