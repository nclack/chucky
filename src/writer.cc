#include <span>
#include <cstddef>
#include <type_traits>
#include <algorithm>
#include <numeric>

namespace stream {

// Result type for write operations - contains unconsumed bytes on success
struct WriteResult {
    std::span<const std::byte> unconsumed;
    
    // Implicit conversion to span for convenience
    operator std::span<const std::byte>() const noexcept {
        return unconsumed;
    }
    
    // Check if all bytes were consumed
    [[nodiscard]] bool all_consumed() const noexcept {
        return unconsumed.empty();
    }
};

// Concept to constrain writer implementations
template<typename T>
concept Writer = requires(T& w, std::span<const std::byte> data) {
    { w.write(data) } -> std::same_as<WriteResult>;
};

// Base writer interface
class WriterInterface {
public:
    virtual ~WriterInterface() = default;
    
    // Main write method - returns unconsumed bytes
    [[nodiscard]] virtual WriteResult write(std::span<const std::byte> data) noexcept = 0;
    
    // Optional: flush any buffered data
    [[nodiscard]] virtual bool flush() noexcept { return true; }
};

// CRTP base class for static polymorphism (optional but useful for composition)
template<typename Derived>
class WriterBase {
public:
    [[nodiscard]] WriteResult write(std::span<const std::byte> data) noexcept {
        return static_cast<Derived*>(this)->write(data);
    }
    
    [[nodiscard]] bool flush() noexcept {
        return static_cast<Derived*>(this)->flush();
    }
};

// Example implementation: Null writer (consumes all bytes)
class NullWriter : public WriterBase<NullWriter> {
public:
    [[nodiscard]] WriteResult write(std::span<const std::byte> /*data*/) noexcept {
        return {{}}; // Return empty span (all consumed)
    }
    
    [[nodiscard]] bool flush() noexcept { return true; }
};

// Example implementation: Limited buffer writer
class BufferWriter : public WriterBase<BufferWriter> {
private:
    std::byte* buffer_;
    std::size_t capacity_;
    std::size_t position_ = 0;

public:
    BufferWriter(std::byte* buffer, std::size_t capacity) 
        : buffer_(buffer), capacity_(capacity) {}
    
    [[nodiscard]] WriteResult write(std::span<const std::byte> data) noexcept {
        if (position_ >= capacity_) {
            return {data}; // Can't write anything, return all as unconsumed
        }
        
        const std::size_t available = capacity_ - position_;
        const std::size_t to_write = std::min(data.size(), available);
        
        std::copy_n(data.begin(), to_write, buffer_ + position_);
        position_ += to_write;
        
        // Return unconsumed portion (if any)
        if (to_write < data.size()) {
            return {data.subspan(to_write)};
        }
        return {{}};
    }
    
    [[nodiscard]] bool flush() noexcept { return true; }
    
    std::size_t written() const noexcept { return position_; }
};

// Composable writer adapter - chains multiple writers
template<Writer Primary, Writer Secondary>
class ChainedWriter : public WriterBase<ChainedWriter<Primary, Secondary>> {
private:
    Primary primary_;
    Secondary secondary_;

public:
    template<typename P, typename S>
    ChainedWriter(P&& primary, S&& secondary)
        : primary_(std::forward<P>(primary))
        , secondary_(std::forward<S>(secondary)) {}
    
    [[nodiscard]] WriteResult write(std::span<const std::byte> data) noexcept {
        // First write to primary
        auto result = primary_.write(data);
        if (result.all_consumed()) {
            return result;
        }
        
        // Write remaining to secondary
        return secondary_.write(result.unconsumed);
    }
    
    [[nodiscard]] bool flush() noexcept {
        return primary_.flush() && secondary_.flush();
    }
};

// Convenience function to create chained writers
template<Writer Primary, Writer Secondary>
auto chain_writers(Primary&& primary, Secondary&& secondary) {
    return ChainedWriter<std::decay_t<Primary>, std::decay_t<Secondary>>(
        std::forward<Primary>(primary),
        std::forward<Secondary>(secondary)
    );
}

// Utility function to convert void* range to span
inline std::span<const std::byte> make_byte_span(const void* beg, const void* end) noexcept {
    const auto size = static_cast<std::size_t>(
        static_cast<const std::byte*>(end) - static_cast<const std::byte*>(beg)
    );
    return std::span{static_cast<const std::byte*>(beg), size};
}

} // namespace stream
