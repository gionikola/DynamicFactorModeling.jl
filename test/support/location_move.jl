module LocationMove

using DynamicFactorModeling

export shift_factor_locations!

# Validation and production call the same implementation. The original
# experimental source is preserved with the completed campaign snapshots.
const shift_factor_locations! = DynamicFactorModeling._shift_factor_locations!

end # module
