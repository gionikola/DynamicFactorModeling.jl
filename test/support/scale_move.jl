module ScaleMove

using DynamicFactorModeling

export rescale_factors!

# Keep the validation interface while testing the actual production kernel.
const rescale_factors! = DynamicFactorModeling._rescale_factors!
const factor_energy = DynamicFactorModeling._mixing_factor_energy
const energy_change = DynamicFactorModeling._mixing_energy_change

end # module
