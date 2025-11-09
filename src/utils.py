def apply_scale(width, height, scale):
    """Ensure dimensions are divisible by 16"""

    def _make_divisible(value, stride):
        return max(stride, int(round(value / stride) * stride))

    new_width = round(width * scale)
    new_height = round(height * scale)
    new_width = _make_divisible(new_width, 16)
    new_height = _make_divisible(new_height, 16)
    return new_width, new_height
