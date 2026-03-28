import numpy as np

class EconCalculations:
    """
    A class to encapsulate and manage economic calculations within the simulation loop.
    
    This class holds the state of economic variables that persist between time periods.
    """
    def __init__(self, econ_params, initial_removal_cost=5000000.0):
        """
        Initializes the economic calculator.
        
        Args:
            econ_params: An object containing economic parameters, like the welfare coefficient.
            initial_removal_cost (float): The cost to remove one piece of debris.
        """
        # --- Parameters ---
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
        
        total_fringe_sat = np.sum(current_environment[fringe_start_slice:fringe_end_slice])
        welfare = 0.5 * self.welfare_coef * total_fringe_sat**2 + welfare_revenue_component
        
        # 3. Update the state for the NEXT period.
        # The new pool of funds is the leftover amount plus the newly collected tax revenue.
        self.total_funds_for_removals = funds_left_before_new_revenue + new_tax_revenue
        
        return welfare, funds_left_before_new_revenue
    
def revenue_open_access_calculations(open_access_inputs, state_next):
    
    # Find the fringe ('Su') species' economic parameters from the solver object
    fringe_econ_params = None
    for sp in open_access_inputs.multi_species.species:
        if sp.name == 'Su':
            fringe_econ_params = sp.econ_params
            break
    
    if fringe_econ_params is None:
        raise ValueError("Could not find 'Su' species econ_params in revenue_open_access_calculations")

    collision_probability = open_access_inputs._last_collision_probability

    fringe_total = state_next[open_access_inputs.fringe_start_slice:open_access_inputs.fringe_end_slice]

    # Use fringe_econ_params instead of open_access_inputs.econ_params
    if (fringe_econ_params.bond is not None) and (fringe_econ_params.bond != 0):
        # Calculate revenue ONLY from sats at end-of-life
        sats_at_eol = fringe_total / fringe_econ_params.sat_lifetime
        
        # Calculate bond revenue: (non-compliance rate) * (sats at EOL) * (bond value)
        revenue_by_shell = (1 - fringe_econ_params.comp_rate) * sats_at_eol * fringe_econ_params.bond
        # Set the attribute on the solver object instead of reading it
        open_access_inputs.bond_revenue = revenue_by_shell                # bond
        open_access_inputs._revenue_type = "bond"

    # Use fringe_econ_params
    elif getattr(fringe_econ_params, "ouf", 0) != 0:
        revenue_by_shell = fringe_econ_params.ouf * fringe_total *collision_probability    # OUF
        open_access_inputs._revenue_type = "ouf"

    # Use fringe_econ_params
    elif fringe_econ_params.tax != 0: #tax
        Cp            = collision_probability
        cost_per_sat  = np.asarray(fringe_econ_params.cost)
        revenue_by_shell = fringe_econ_params.tax * Cp * cost_per_sat * fringe_total
        open_access_inputs._revenue_type = "tax"

    else:  # nothing levied
        revenue_by_shell = np.zeros_like(fringe_total)
        open_access_inputs._revenue_type = "none"

    total_revenue = revenue_by_shell.sum()

    _last_tax_revenue   = revenue_by_shell
    _last_total_revenue = float(total_revenue)
    _dbg_tax_rate       = fringe_econ_params.tax
    _dbg_Cp             = collision_probability  
    _dbg_cost_per_sat   = np.asarray(fringe_econ_params.cost)
    _dbg_fringe_total   = fringe_total

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