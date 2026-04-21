import numpy as np

class EconCalculations:
    """
    A class to encapsulate and manage economic calculations within the simulation loop.
    
    This class holds the state of economic variables that persist between time periods.
    """
    def __init__(self, econ_params, initial_removal_cost=5000000.0, welfare_coef=None,
                 welfare_species=None):
        """
        Initializes the economic calculator.
        
        Args:
            econ_params: An object containing economic parameters. Used only for removal_cost
                and as a fallback source of `coef` if welfare_coef is not provided.
            initial_removal_cost (float): The cost to remove one piece of debris.
            welfare_coef (float): Fallback welfare coefficient used in single-species
                welfare mode. Kept for backward compatibility. Ignored if
                welfare_species is provided.
            welfare_species (list of (name, start_slice, end_slice, coef) tuples):
                Per-species entries to include in the welfare calculation. Each
                species contributes 0.5 * coef * S_species^2 to consumer surplus,
                summed across species. When provided, this takes precedence over
                welfare_coef. Typical usage: include both S and Su since both are
                endogenous launch species under the updated paper.
        """
        # --- Parameters ---
        self.welfare_species = welfare_species
        if welfare_coef is not None:
            self.welfare_coef = welfare_coef
        else:
            self.welfare_coef = econ_params.coef
        self.removal_cost = initial_removal_cost

        # --- State Variable ---
        self.total_funds_for_removals = 0.0

    def get_removals_for_current_period(self):
        """
        Calculates how many removals can be afforded in the current period
        based on the *total available funds*.
        
        Returns:
            int: The number of affordable removals.
        """
        if self.removal_cost > 0:
            return int(self.total_funds_for_removals // self.removal_cost)
        return 0

    def process_period_economics(self, num_actually_removed, current_environment, fringe_slices, new_tax_revenue):
        """
        Performs all economic calculations for the current period and updates the state for the next.
        
        Args:
            num_actually_removed (int): The number of objects removed in this period.
            current_environment (np.ndarray): The state of the orbital environment.
            fringe_slices (tuple): A tuple (start, end) for the fringe satellite slice.
            new_tax_revenue (float): The total tax revenue generated in this period.
            
        Returns:
            tuple: A tuple containing (welfare, funds_left_before_new_revenue).
        """
        fringe_start_slice, fringe_end_slice = fringe_slices

        # 1. Calculate the cost of this period's removals and the remaining funds
        cost_of_removals = num_actually_removed * self.removal_cost
        funds_left_before_new_revenue = self.total_funds_for_removals - cost_of_removals
        
        # 2. Calculate welfare for this period.
        # The welfare bonus is based on the unspent funds from the available budget.
        welfare_revenue_component = max(0, funds_left_before_new_revenue)
        
        # Consumer surplus summed across all endogenous species that carry their
        # own demand curve. Falls back to single-species (fringe-only) mode if 
        # welfare_species wasn't provided at construction.
        if self.welfare_species:
            consumer_surplus = 0.0
            for (_name, start, end, coef) in self.welfare_species:
                sp_pop = np.sum(current_environment[start:end])
                consumer_surplus += 0.5 * coef * sp_pop ** 2
        else:
            total_fringe_sat = np.sum(current_environment[fringe_start_slice:fringe_end_slice])
            consumer_surplus = 0.5 * self.welfare_coef * total_fringe_sat ** 2

        welfare = consumer_surplus + welfare_revenue_component
        
        # 3. Update the state for the NEXT period.
        # The new pool of funds is the leftover amount plus the newly collected tax revenue.
        self.total_funds_for_removals = funds_left_before_new_revenue + new_tax_revenue
        
        return welfare, funds_left_before_new_revenue
    
def revenue_open_access_calculations(open_access_inputs, state_next):
    """
    Compute policy revenue aggregated across all active (launching) species.

    Per the updated paper, tax/bond/OUF apply to all active satellite species
    that participate in the open-access launch equilibrium, not only the
    fringe (Su). Revenue is summed over species so that each satellite in
    orbit, regardless of species, contributes to the funding pool for ADR.

    The revenue TYPE (bond vs OUF vs tax) is chosen once, based on what policy
    is active. If more than one is active, priority is bond > OUF > tax (only
    the highest-priority active policy contributes). This matches the previous
    per-species logic and reflects that the three instruments are alternatives,
    not additive. The per-species parameter values are taken from each
    species' own econ_params, so it is possible (though not typical) for
    species to carry different bond/tax/OUF values; the same policy "type"
    is applied to all of them.
    """
    # Only these species participate in open-access launches and thus in the
    # revenue pool. Extend this list if more species become endogenous.
    ACTIVE_SPECIES = ('S', 'Su')

    cp_dict = open_access_inputs._last_collision_probability or {}
    n_shells = open_access_inputs.MOCAT.scenario_properties.n_shells

    # Determine policy type from any active species' econ_params. All active
    # species share the same policy flags in the current grid setup (policy
    # values are assigned uniformly in optimize_ADR.solve_year_zero), so any
    # species' econ_params is a valid source; prefer Su for backward
    # compatibility with scenarios that only set it on Su.
    policy_reference = None
    for sp in open_access_inputs.multi_species.species:
        if sp.name in ACTIVE_SPECIES:
            if sp.name == 'Su' or policy_reference is None:
                policy_reference = sp.econ_params
                if sp.name == 'Su':
                    break

    if policy_reference is None:
        raise ValueError(
            "No active species (S or Su) found in revenue_open_access_calculations"
        )

    # Initialize aggregate revenue and per-species bond revenue (needed by
    # downstream consumers that expect a dict of bond revenue).
    revenue_by_shell_total = np.zeros(n_shells)
    bond_revenue_by_shell = np.zeros(n_shells)

    # Select policy type once, then iterate over species. Priority: bond > OUF > tax.
    bond_active = (policy_reference.bond is not None) and (policy_reference.bond != 0)
    ouf_active  = getattr(policy_reference, 'ouf', 0) != 0
    tax_active  = policy_reference.tax != 0

    if bond_active:
        revenue_type = "bond"
    elif ouf_active:
        revenue_type = "ouf"
    elif tax_active:
        revenue_type = "tax"
    else:
        revenue_type = "none"

    for sp in open_access_inputs.multi_species.species:
        if sp.name not in ACTIVE_SPECIES:
            continue

        sp_ep = sp.econ_params
        sp_pop = state_next[sp.start_slice:sp.end_slice]
        sp_cp = cp_dict.get(sp.name, np.zeros_like(sp_pop))

        if revenue_type == "bond" and (sp_ep.bond is not None) and (sp_ep.bond != 0):
            sats_at_eol = sp_pop / sp_ep.sat_lifetime
            sp_revenue = (1 - sp_ep.comp_rate) * sats_at_eol * sp_ep.bond
            bond_revenue_by_shell += sp_revenue
            revenue_by_shell_total += sp_revenue

        elif revenue_type == "ouf" and getattr(sp_ep, 'ouf', 0) != 0:
            sp_revenue = sp_ep.ouf * sp_pop * sp_cp
            revenue_by_shell_total += sp_revenue

        elif revenue_type == "tax" and sp_ep.tax != 0:
            cost_per_sat = np.asarray(sp_ep.cost)
            sp_revenue = sp_ep.tax * sp_cp * cost_per_sat * sp_pop
            revenue_by_shell_total += sp_revenue
        # else: this species isn't carrying this policy, contributes 0

    open_access_inputs.bond_revenue = bond_revenue_by_shell
    open_access_inputs._revenue_type = revenue_type

    total_revenue = revenue_by_shell_total.sum()

    _last_tax_revenue   = revenue_by_shell_total
    _last_total_revenue = float(total_revenue)
    # Keep the debug returns the same shape as before so callers don't break.
    # Reference values are from the policy_reference species (usually Su).
    _dbg_tax_rate       = policy_reference.tax
    _dbg_Cp             = cp_dict.get('Su', np.zeros(n_shells))
    _dbg_cost_per_sat   = np.asarray(policy_reference.cost)
    _dbg_fringe_total   = state_next[open_access_inputs.fringe_start_slice:open_access_inputs.fringe_end_slice]

    return _last_tax_revenue, _last_total_revenue, _dbg_tax_rate, _dbg_Cp, _dbg_cost_per_sat, _dbg_fringe_total

# def calibrate_static_maneuver_price(current_environment, mocat, multi_species, elliptical, target_annual_cost=100000.0):
#     """
#     Calculates a static price-per-maneuver based strictly on Year 0 population distribution.
#     This anchors the price, preventing economic feedback loops in the solver.
#     """
#     maneuver_prices = {}

#     # Format the environment array for the MOCAT functions
#     env_flat = current_environment.flatten() if elliptical else current_environment

#     for species in multi_species.species:
#         if not species.maneuverable:
#             maneuver_prices[species.name] = 0.0
#             continue

#         # 1. Get initial population for this species
#         if elliptical:
#             initial_pop = current_environment[:, species.species_idx, 0]
#         else:
#             initial_pop = current_environment[species.start_slice:species.end_slice]

#         # 2. Get initial physical maneuvers expected at Year 0
#         evaluated_value = mocat.scenario_properties.fringe_active_loss['maneuvers'][species.name](*env_flat)
#         initial_maneuvers = np.array([float(value[0]) for value in evaluated_value])

#         total_sats = np.sum(initial_pop)

#         if total_sats <= 0:
#             maneuver_prices[species.name] = 0.0
#             continue

#         # 3. Total maneuvers experienced by the whole fleet
#         total_fleet_maneuvers = np.sum(initial_pop * initial_maneuvers)

#         # 4. Population-weighted average maneuvers for a single satellite
#         weighted_avg_maneuvers = total_fleet_maneuvers / total_sats

#         # 5. Calculate the static price multiplier
#         if weighted_avg_maneuvers > 0:
#             static_multiplier = target_annual_cost / weighted_avg_maneuvers
#         else:
#             static_multiplier = 0.0
            
#         maneuver_prices[species.name] = static_multiplier

#         print(f"\n--- Year 0 Calibration for {species.name} ---")
#         print(f"Total Fleet Pop: {total_sats:.0f}")
#         print(f"Weighted Avg Maneuvers: {weighted_avg_maneuvers:.2f}")
#         print(f"Static Price per Maneuver: ${static_multiplier:,.2f}")

#     return maneuver_prices

    
def calibrate_static_maneuver_price(current_environment, mocat, multi_species, elliptical, target_annual_cost=100000.0):
    """
    Calculates a static price-per-maneuver based strictly on Year 0 population distribution
    aggregated across ALL maneuverable species. This ensures the "average maneuverable
    satellite" across the entire environment pays the target annual cost.
    """
    maneuver_prices = {}

    # Format the environment array for the MOCAT functions
    env_flat = current_environment.flatten() if elliptical else current_environment

    global_total_sats = 0.0
    global_total_maneuvers = 0.0

    # --- PASS 1: Aggregate totals across all maneuverable species ---
    for species in multi_species.species:
        if not species.maneuverable:
            continue

        # Get initial population for this species
        if elliptical:
            initial_pop = current_environment[:, species.species_idx, 0]
        else:
            initial_pop = current_environment[species.start_slice:species.end_slice]

        # Get initial physical maneuvers expected at Year 0
        evaluated_value = mocat.scenario_properties.fringe_active_loss['maneuvers'][species.name](*env_flat)
        initial_maneuvers = np.array([float(value[0]) for value in evaluated_value])

        # Add to global counts
        global_total_sats += np.sum(initial_pop)
        global_total_maneuvers += np.sum(initial_pop * initial_maneuvers)

    # --- CALCULATION: Determine the single global multiplier ---
    if global_total_sats > 0 and global_total_maneuvers > 0:
        global_weighted_avg = global_total_maneuvers / global_total_sats
        global_static_multiplier = target_annual_cost / global_weighted_avg
    else:
        global_weighted_avg = 0.0
        global_static_multiplier = 0.0

    print(f"\n--- Year 0 Global Calibration (All Maneuverable Species) ---")
    print(f"Global Fleet Pop: {global_total_sats:.0f}")
    print(f"Global Weighted Avg Maneuvers: {global_weighted_avg:.2f}")
    print(f"Global Static Price per Maneuver: ${global_static_multiplier:,.2f}")

    # --- PASS 2: Assign the global multiplier back to the dictionary ---
    for species in multi_species.species:
        if species.maneuverable:
            maneuver_prices[species.name] = global_static_multiplier
        else:
            maneuver_prices[species.name] = 0.0

    return maneuver_prices