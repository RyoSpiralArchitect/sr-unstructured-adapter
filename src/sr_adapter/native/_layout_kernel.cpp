#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
struct LayoutBox {
    double x0;
    double y0;
    double x1;
    double y1;
    double score;
    std::int32_t page;
    std::int32_t order_hint;
};

struct LayoutResult {
    std::int32_t original_index;
    std::int32_t order;
    std::int32_t page;
    std::int32_t label;
    double confidence;
    double x_center;
    double y_center;
};

constexpr std::int32_t LABEL_PARAGRAPH = 0;
constexpr std::int32_t LABEL_HEADING = 1;
constexpr std::int32_t LABEL_TABLE = 2;
constexpr std::int32_t LABEL_FIGURE = 3;

inline double clamp(double value, double low, double high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

inline double area(const LayoutBox &box) {
    const double w = std::max(0.0, box.x1 - box.x0);
    const double h = std::max(0.0, box.y1 - box.y0);
    return w * h;
}

inline std::int32_t classify(const LayoutBox &box, double threshold) {
    const double w = std::max(0.001, box.x1 - box.x0);
    const double h = std::max(0.001, box.y1 - box.y0);
    const double ar = w / h;
    const double a = area(box);

    if (a < threshold * 0.25 || h < 10.0) {
        return LABEL_HEADING;
    }
    if (ar > 3.0 && h > 20.0) {
        return LABEL_TABLE;
    }
    if (a > threshold * 4.0) {
        return LABEL_FIGURE;
    }
    return LABEL_PARAGRAPH;
}

inline LayoutResult to_result(const LayoutBox &box, std::int32_t original_index,
                              std::int32_t order, double threshold) {
    LayoutResult result{};
    result.original_index = original_index;
    result.order = order;
    result.page = box.page;
    result.label = classify(box, threshold);
    const double w = std::max(0.001, box.x1 - box.x0);
    const double h = std::max(0.001, box.y1 - box.y0);
    const double norm_score = clamp(box.score, 0.0, 1.0);
    const double balance = clamp(w / (h + 0.001), 0.0, 6.0);
    double confidence = threshold * 0.35 + norm_score * 0.5 + balance * 0.03;
    if (result.label == LABEL_HEADING) {
        confidence += 0.05;
    } else if (result.label == LABEL_TABLE) {
        confidence += 0.1;
    } else if (result.label == LABEL_FIGURE) {
        confidence -= 0.05;
    }
    result.confidence = clamp(confidence, 0.01, 0.99);
    result.x_center = (box.x0 + box.x1) * 0.5;
    result.y_center = (box.y0 + box.y1) * 0.5;
    return result;
}
} // namespace

extern "C" {
struct LayoutBoxC {
    double x0;
    double y0;
    double x1;
    double y1;
    double score;
    std::int32_t page;
    std::int32_t order_hint;
};

struct LayoutResultC {
    std::int32_t original_index;
    std::int32_t order;
    std::int32_t page;
    std::int32_t label;
    double confidence;
    double x_center;
    double y_center;
};

int analyze_layout(const LayoutBoxC *boxes, std::int32_t count, double threshold,
                   LayoutResultC *results) {
    if (!boxes || !results || count <= 0) {
        return 0;
    }

    std::vector<std::pair<LayoutBox, std::int32_t>> items;
    items.reserve(count);
    for (std::int32_t i = 0; i < count; ++i) {
        LayoutBox box{boxes[i].x0, boxes[i].y0, boxes[i].x1, boxes[i].y1, boxes[i].score,
                      boxes[i].page, boxes[i].order_hint};
        items.emplace_back(box, i);
    }

    std::stable_sort(items.begin(), items.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.first.page != rhs.first.page) {
            return lhs.first.page < rhs.first.page;
        }
        const double ly = lhs.first.y0;
        const double ry = rhs.first.y0;
        if (std::fabs(ly - ry) > 1e-3) {
            return ly < ry;
        }
        if (lhs.first.order_hint != rhs.first.order_hint) {
            return lhs.first.order_hint < rhs.first.order_hint;
        }
        return lhs.first.x0 < rhs.first.x0;
    });

    std::int32_t order = 0;
    for (const auto &entry : items) {
        const LayoutBox &box = entry.first;
        const std::int32_t original_index = entry.second;
        LayoutResult result = to_result(box, original_index, order++, threshold);
        results[order - 1] = {result.original_index, result.order, result.page, result.label,
                              result.confidence, result.x_center, result.y_center};
    }

    return order;
}

double calibrate_threshold(const double *scores, std::int32_t count, double current_threshold) {
    if (!scores || count <= 0) {
        return current_threshold;
    }
    double sum = 0.0;
    double min_score = scores[0];
    for (std::int32_t i = 0; i < count; ++i) {
        double v = scores[i];
        sum += v;
        if (v < min_score) {
            min_score = v;
        }
    }
    const double mean = sum / static_cast<double>(count);
    double updated = current_threshold;
    if (min_score < current_threshold) {
        updated = (mean * 0.6) + (min_score * 0.4);
    } else {
        updated = mean * 0.75 + current_threshold * 0.25;
    }
    return clamp(updated + 0.05, 0.05, 0.95);
}
}
