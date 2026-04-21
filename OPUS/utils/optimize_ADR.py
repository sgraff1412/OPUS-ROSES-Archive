from utils.ConstellationParameters import ConstellationParameters
from utils.EconParameters import EconParameters
from utils.MocatParameters import configure_mocat   
from utils.OpenAccessSolver import OpenAccessSolver
from utils.PostProcessing import PostProcessing
from utils.PlotHandler import PlotHandler
from utils.PostMissionDisposal import evaluate_pmd, evaluate_pmd_elliptical
from utils.MultiSpecies import MultiSpecies
from utils.MultiSpeciesOpenAccessSolver import MultiSpeciesOpenAccessSolver
from utils.Helpers import insert_launches_into_lam
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import time
from datetime import timedelta
import os
import pandas as pd

from utils.ADRParameters import ADRParameters
from utils.ADR import optimize_ADR_removal, implement_adr
from utils.EconCalculations import EconCalculations, calibrate_static_maneuver_price
from itertools import repeat


# Species that participate in consumer-surplus aggregation. Per the updated
# paper, both S (constellation) and Su (fringe) are endogenous launch species,
# so both contribute to welfare. If more species become endogenous later,
# extend this set.
WELFARE_ACTIVE_SPECIES = ('S', 'Su')


def _build_welfare_species(multi_species):
    """
    Build the welfare_species list consumed by EconCalculations from the 
    multi_species container. Each entry is (name, start_slice, end_slice, coef)
    where coef is pulled from the species' own econ_params.
    """
    entries = []
    for sp in multi_species.species:
        if sp.name in WELFARE_ACTIVE_SPECIES:
            entries.append((sp.name, sp.start_slice, sp.end_slice, sp.econ_params.coef))
    return entries





class OptimizeADR:
    def __init__(self, params = []):
        """
            Initializing the OptimizeADR Class.
            Used for greedy optimization of ADR during each timestep. 
        """
        self.output = None
        self.MOCAT = None
        self.econ_params_json = None
        self.pmd_linked_species = None
        self.adr_params_json = None
        self.params = params 
        self.umpy_score = None
        self.welfare_dict = {}
        self.adr_dict = {}
        self.config = None
        self.target_annual_maneuver_cost = 100000.0

    @staticmethod
    def optimizer_get_species_position_indexes(MOCAT, multi_species):
        """
            The MOCAT model works on arrays that are the number of shells x number of species.
            Often throughout the model, we see the original list being spliced. 
            This function returns the start and end slice of the species in the array.

            Inputs:
                MOCAT: The MOCAT model
                constellation_sats: The name of the constellation satellites
                fringe_sats: The name of the fringe satellites
        """
        for i, sp in enumerate(multi_species.species):
            if sp.name == 'S':
                constellation_sats_idx = sp.species_idx
                constellation_start_slice = sp.start_slice
                constellation_end_slice = sp.end_slice

            if sp.name == 'Su':
                fringe_sats_idx = sp.species_idx
                fringe_start_slice = sp.start_slice
                fringe_end_slice = sp.end_slice

        return constellation_start_slice, constellation_end_slice, fringe_start_slice, fringe_end_slice

    def solve_year_zero(self, scenario_name, MOCAT_config, simulation_name, grid_search=False):
        """
            Sets up the initial environment. 
        """
        self.grid_search = grid_search
        multi_species_names = ["S", "Su"]
        multi_species = MultiSpecies(multi_species_names)

        #########################
        ### CONFIGURE MOCAT MODEL
        #########################
        self.MOCAT, multi_species = configure_mocat(MOCAT_config, multi_species=multi_species, grid_search=self.grid_search)
        self.elliptical = self.MOCAT.scenario_properties.elliptical # elp, x0 = 12517
        print(self.MOCAT.scenario_properties.x0)

        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1)
        # create a list of the years (int) using scenario_properties.start_date and scenario_properties.simulation_duration
        years = [int(self.MOCAT.scenario_properties.start_date.year) + i for i in range(self.MOCAT.scenario_properties.simulation_duration)]
        years.insert(0, years[0] - 1)

        multi_species.get_species_position_indexes(self.MOCAT)
        multi_species.get_mocat_species_parameters(self.MOCAT) # abstract species level information, like deltat, etc. 
        current_environment = self.MOCAT.scenario_properties.x0 # Starts as initial population, and is in then updated. 
        species_data = {sp: {year: np.zeros(self.MOCAT.scenario_properties.n_shells) for year in years} for sp in self.MOCAT.scenario_properties.species_names}

        # update time 0 as the initial population
        initial_year = years[0] # Get the first year (e.g., 2016)

        if self.elliptical:
            x0_alt = self.MOCAT.scenario_properties.sma_ecc_mat_to_altitude_mat(self.MOCAT.scenario_properties.x0)
        
        for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
            if self.elliptical:
                species_data[sp][initial_year] = x0_alt[:, i]
            else:
                species_data[sp][initial_year] = self.MOCAT.scenario_properties.x0[sp]

        _, _, fringe_start_slice, fringe_end_slice = self.optimizer_get_species_position_indexes(MOCAT=self.MOCAT, multi_species=multi_species)

        shells = np.arange(1, self.MOCAT.scenario_properties.n_shells +1)
        for sp in MOCAT_config['species']:
            if "OPUS" in sp:
                # Determine naturally compliant shell for 5 and 25 year PMD rules
                if sp['OPUS']['disposal_time'] == 5:
                    mids = self.MOCAT.scenario_properties.HMid
                    mids = [abs(x - 400) for x in mids]
                    natrually_compliant_idx = mids.index(min(mids))
                    not_naturally_compliant_shells = shells[(natrually_compliant_idx+1):-1]
                elif sp['OPUS']['disposal_time'] == 25:
                    mids = self.MOCAT.scenario_properties.HMid
                    mids = [abs(x - 520) for x in mids]
                    natrually_compliant_idx = mids.index(min(mids))
                    not_naturally_compliant_shells = shells[(natrually_compliant_idx+1):-1]

        #################################
        ### CONFIGURE ECONOMIC PARAMETERS
        #################################
        econ_params = EconParameters(self.econ_params_json, mocat=self.MOCAT)

        # J-Rewrote the below
        # Check if a parameter grid is being used for the current scenario
        current_params = None
        if self.params is not None and len(self.params) > 0:
            # Find the parameters for the current scenario_name
            for p in self.params:
                if p[0] == scenario_name:
                    current_params = p
                    break

        if current_params:
            # If parameters are found in the grid, apply them
            econ_params.removal_cost = float(current_params[4])
            econ_params.tax = float(current_params[5])
            econ_params.bond = float(current_params[6]) if current_params[6] is not None else None
            if econ_params.bond == 0:
                econ_params.bond = None
            econ_params.ouf = float(current_params[7])

            if len(current_params) > 10:
                econ_params.disposal_time = float(current_params[10])

            # Pick up maneuver cost from params grid if provided (index 11).
            # This overrides the default self.target_annual_maneuver_cost (100000)
            # and is used later in calibrate_static_maneuver_price.
            if len(current_params) > 11 and current_params[11] is not None:
                self.target_annual_maneuver_cost = float(current_params[11])

            for species in multi_species.species:
                species.econ_params.ouf = getattr(econ_params, 'ouf', 0.0)
                species.econ_params.bond = econ_params.bond
                species.econ_params.tax = econ_params.tax

                if len(current_params) > 10:
                    species.econ_params.disposal_time = float(current_params[10])
                    
                species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name) 
            self.econ_params = econ_params
            # Welfare sums consumer surplus across all endogenous species (both S and Su).
            # Each species contributes 0.5 * coef * S_species^2 where coef is that species'
            # demand coefficient and S_species is its per-period population.
            welfare_species_list = _build_welfare_species(multi_species)
            econ_calculator = EconCalculations(self.econ_params, initial_removal_cost=5000000,
                                                welfare_species=welfare_species_list)

            # econ_params.calculate_cost_fn_parameters()
            # sammie / joey: i think we need the above line to run this but i'm not sure what the inputs would be
        else:
            self.econ_params = EconParameters(self.econ_params_json, mocat=self.MOCAT)
            self.econ_params.econ_params_for_ADR(scenario_name)
            # For each simulation - we will need to modify the base economic parameters for the species. 
            for species in multi_species.species:
                species.econ_params.modify_params_for_simulation(scenario_name)
                species.econ_params.calculate_cost_fn_parameters(species.Pm, scenario_name)            
            # species.econ_params.update_congestion_costs(multi_species, self.MOCAT.scenario_properties.x0)
            welfare_species_list = _build_welfare_species(multi_species)
            econ_calculator = EconCalculations(self.econ_params, initial_removal_cost=5000000,
                                                welfare_species=welfare_species_list)

        # For now make all satellites circular if elliptical
        if self.elliptical:
            # for species idx in multi_species.species, place all of their satellites in the first eccentricity bin
            for species in multi_species.species:
                # Sum all satellites across all eccentricity bins for this species
                total_satellites = np.sum(self.MOCAT.scenario_properties.x0[:, species.species_idx, :], axis=1)
                # Move all satellites to the first eccentricity bin (index 0)
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 0] = total_satellites
                # Set all other eccentricity bins to zero
                self.MOCAT.scenario_properties.x0[:, species.species_idx, 1:] = 0

        # Flatten for circular orbits
        if not self.elliptical:     
            self.MOCAT.scenario_properties.x0 = self.MOCAT.scenario_properties.x0.T.values.flatten()           

        # Set current environment as initial conditions from MOCAT
        current_environment = self.MOCAT.scenario_properties.x0

        # Maneuvering costs
        self.static_maneuver_prices = calibrate_static_maneuver_price(
            current_environment=current_environment,
            mocat=self.MOCAT,
            multi_species=multi_species,
            elliptical=self.elliptical,
            target_annual_cost=self.target_annual_maneuver_cost
        )          

        # Set up  parameters for ADR, set ADR times as every year after first year
        self.adr_params = ADRParameters(self.adr_params_json, mocat=self.MOCAT)
        self.adr_params.adr_parameter_setup(scenario_name)
        self.adr_params.adr_times = np.arange(2, self.MOCAT.scenario_properties.simulation_duration+1)
            
        
        if current_params:
        # If parameters are found in the grid, apply them
            self.adr_params.target_species = [current_params[1]]
            self.adr_params.target_shell = [current_params[2]]
            self.adr_params.shell_order = [12, 14, 13, 15, 17, 11, 18, 16, 19, 20, 10, 9, 8, 5, 6, 7, 4, 3, 2, 1] # set as initial shell order; may update later
            # ADR is endogenous if ANY revenue-raising policy is active:
            # tax, bond, or OUF. Exogenous means no policy-driven funding
            # (ADR funded externally with hardcoded 10 removals/year as a stand-in).
            # Previously this required BOTH bond AND ouf simultaneously, which 
            # disabled endogenous mode for tax-only, bond-only, and OUF-only 
            # scenarios.
            tax_active  = (econ_params.tax is not None) and (econ_params.tax != 0)
            bond_active = (econ_params.bond is not None) and (econ_params.bond != 0)
            ouf_active  = (econ_params.ouf is not None) and (econ_params.ouf != 0)
            if tax_active or bond_active or ouf_active:
                self.adr_params.exogenous = 0
            else:
                self.adr_params.exogenous = 1
            if current_params[3] > 1:
                self.adr_params.n_remove = [current_params[3]] 
                self.adr_params.remove_method = ["n"]
            elif current_params[3] < 1:
                self.adr_params.p_remove = [current_params[3]]
                self.adr_params.remove_method = ["p"]

        # Solver guess is 5% of the current fringe satellites. Update The launch file. This essentially helps the optimiser, as it is not a random guess to start with. 
        # Lam should be the same shape as x0 and is full of None values for objects that are not launched. 
        solver_guess = self.MOCAT.scenario_properties.x0.copy()
        lam = np.full_like(self.MOCAT.scenario_properties.x0, None, dtype=object)
        if self.elliptical:
            for species in multi_species.species:
                # lam will be n_sma_bins x n_ecc_bins x n_alt_shells
                initial_guess = 0.05 * self.MOCAT.scenario_properties.x0[:, species.species_idx, 0]
                # if sum of initial guess is 0, multiply each element by 10
                if np.sum(initial_guess) == 0:
                    initial_guess[:] = 5
                lam[:, species.species_idx, 0] = initial_guess
                solver_guess[:, species.species_idx, 0] = initial_guess
        else:
            for species in multi_species.species:
                # if species.name == constellation_sat:
                #     continue
                # else:
                inital_guess = 0.05 * np.array(self.MOCAT.scenario_properties.x0[species.start_slice:species.end_slice])  
                # if sum of initial guess is 0, muliply each element by 10
                if sum(inital_guess) == 0:
                    inital_guess[:] = 5
                solver_guess[species.start_slice:species.end_slice] = inital_guess
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]

        # solver_guess = self._apply_replacement_floor(solver_guess, self.MOCAT.scenario_properties.x0, multi_species)
        if self.elliptical:
            for species in multi_species.species:
                lam[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0]
        else:
            for species in multi_species.species:
                lam[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice]
        # for species in multi_species.species:
        #     if species.adr_params is None:
        #         species.adr_params.adr_parameter_setup(scenario_name)

        ############################
        ### SOLVE FOR THE FIRST YEAR
        ############################c
        open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, self.MOCAT.scenario_properties.x0, "linear", lam, multi_species, years, 0, fringe_start_slice, fringe_end_slice, static_maneuver_prices=self.static_maneuver_prices)

        # This is now the first year estimate for the number of fringe satellites that should be launched.
        launch_rate = open_access.solver()
        # launch rate is 92
        # launch rate is 6075

        # sammie/ joey addition: This populates the `total_funds_for_removals` available for the start of the simulation loop (Year 1).
        econ_calculator.process_period_economics(
            num_actually_removed=0,
            current_environment=self.MOCAT.scenario_properties.x0,
            fringe_slices=(fringe_start_slice, fringe_end_slice),
            new_tax_revenue=float(open_access._last_total_revenue)
        )

        lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)
             
        ####################
        ### SIMULATION LOOP
        # For each year, take the previous state of the environment, 
        # then use the sovler to calculate the optimal launch rate. 
        ####################
        model_horizon = self.MOCAT.scenario_properties.simulation_duration
        tf = np.arange(1, model_horizon + 1) 
        self.scenario_name = scenario_name

        return tf, years, current_environment, multi_species, species_data, econ_calculator, shells, lam, fringe_start_slice, fringe_end_slice

    def optimize_adr_loop(self, years, time_idx, multi_species, species_data, econ_calculator, shells, current_environment, lam, fringe_start_slice, fringe_end_slice):
        """
            Propagate the environment and calculate economic and ADR outputs 
            through use of a greedy optimization loop.
        """
        current_trial_results = {}
        try:
            print("Starting year ", years[time_idx-1])
        except Exception as e:
            print("Starting year ", time_idx)
        # tspan = np.linspace(tf[time_idx], tf[time_idx + 1], time_step) # simulate for one year 
        tspan = np.linspace(0, 1, 2)
        shell_welfare = []
        # Propagate the model and take the final state of the environment
        if self.elliptical:
            state_next_sma, state_next_alt = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical, use_euler=True, step_size=0.01)
        else:
            state_next_path, _ = self.MOCAT.propagate(tspan, current_environment, lam, elliptical=self.elliptical) # state_next_path: circ = 12077 elp = alt = 17763, self.x0: circ = 17914, elp = 17914
            if len(state_next_path) > 1:
                state_next_alt = state_next_path[-1, :]
            else:
                state_next_alt = state_next_path 

        # Apply PMD (Post Mission Disposal) evaluation to remove satellites
        print(f"Before PMD - Total environment: {np.sum(state_next_alt)}")
        if self.elliptical:
                # c heck if density_model has name property
            if self.MOCAT.scenario_properties.density_model != "static_exp_dens_func":
                try:
                    density_model_name = self.MOCAT.scenario_properties.density_model.__name__
                except AttributeError:
                    raise ValueError(f"Density model {self.MOCAT.scenario_properties.density_model} does not have a name property")
            state_next_sma, state_next_alt, multi_species = evaluate_pmd_elliptical(state_next_sma, state_next_alt, multi_species, 
                years[time_idx-1], density_model_name, self.MOCAT.scenario_properties.HMid, self.MOCAT.scenario_properties.eccentricity_bins, 
                self.MOCAT.scenario_properties.R0_rad_km)
        else:
            state_next_alt, multi_species = evaluate_pmd(state_next_alt, multi_species)
        print(f"After PMD - Total environment: {np.sum(state_next_alt)}")

        environment_for_solver = state_next_sma if self.elliptical else state_next_alt

        # # ----- ADR Section ---- # #
        # determine removals possible based on cost of removal and money available
        removals_possible = econ_calculator.get_removals_for_current_period()
        # save removals left based on whether ADR is exogenous or endogenous
        # Policy activity is checked cleanly: bond=None or 0 both count as inactive.
        tax_active  = (self.econ_params.tax is not None) and (self.econ_params.tax != 0)
        bond_active = (self.econ_params.bond is not None) and (self.econ_params.bond != 0)
        ouf_active  = (self.econ_params.ouf is not None) and (self.econ_params.ouf != 0)

        if self.adr_params.exogenous == 1:
            self.adr_params.removals_left = 10
        elif not (tax_active or bond_active or ouf_active):
            # No policy active and not exogenous — no funding, no removals.
            self.adr_params.removals_left = 0
        else:
            # Endogenous mode with at least one active policy; use the funds pool.
            self.adr_params.removals_left = removals_possible
        self.adr_params.time = time_idx
        
        # save copy of environment before any ADR occurs
        before_adr = environment_for_solver.copy()
        self.environment_before_adr = before_adr.copy()
        scenario_name = self.scenario_name
        lam_before_adr = lam.copy()

        # If this is a Baseline scenario (no ADR target species, or zero amount to 
        # remove), looping over every shell is wasteful because each trial produces 
        # identical results — optimize_ADR_removal is a no-op and the solver sees 
        # the same input every time. Run a single trial in that case.
        target_species_list = self.adr_params.target_species or []
        is_baseline_scenario = (
            len(target_species_list) == 0
            or (len(target_species_list) == 1 and target_species_list[0] == "none")
        )
        shells_to_iterate = [shells[0]] if is_baseline_scenario else shells

        # run loop for each shell in the model
        for cs in shells_to_iterate:
            # reset environment and removals_left
            trial_environment_for_solver = before_adr.copy()
            # self.adr_params.removals_left = removals_possible
            self.adr_params.target_shell = [cs]

            if ((self.adr_params.adr_times is not None) and (time_idx in self.adr_params.adr_times) and (len(self.adr_params.adr_times) != 0)):
                # environment_for_solver, ~ = implement_adr(environment_for_solver,self.MOCAT,adr_params)
                trial_environment_for_solver, removal_dict = optimize_ADR_removal(trial_environment_for_solver,self.MOCAT,self.adr_params)
            else:
                removal_dict = {'No ADR'}

            # find ADR outputs for the trial
            trial_num_removed = int(np.sum(before_adr - trial_environment_for_solver))
            trial_cost_of_removals = trial_num_removed * econ_calculator.removal_cost
            trial_funds_left = econ_calculator.total_funds_for_removals - trial_cost_of_removals
            trial_leftover_tax_revenue = max(0, trial_funds_left)

            # Record propagated environment data 
            for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                # 0 based index 
                if self.elliptical:
                    # For elliptical orbits, propagated_environment is a 2D array (n_shells, n_species)
                    species_data[sp][years[time_idx]] = state_next_alt[:, i]
                else:
                    # For circular orbits, propagated_environment is a 1D array
                    species_data[sp][years[time_idx]] = state_next_alt[i * self.MOCAT.scenario_properties.n_shells:(i + 1) * self.MOCAT.scenario_properties.n_shells]

            # Fringe Equilibrium Controller
            start_time = time.time()
            # solver guess will be lam
            solver_guess = lam_before_adr.copy()
            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice,static_maneuver_prices=self.static_maneuver_prices)

            # Calculate solver_guess
            solver_guess = lam_before_adr.copy()
            for species in multi_species.species:
                # Calculate the probability of collision based on the new position
                collision_probability = open_access.calculate_probability_of_collision(state_next_alt, species.name)

                if species.maneuverable:
                    maneuvers = open_access.calculate_maneuvers(state_next_alt, species.name)
                    
                    # --- APPLYING STATIC MANEUVER COSTS ---
                    cost_multiplier = self.static_maneuver_prices.get(species.name, 0.0)
                    maneuver_cost = maneuvers * cost_multiplier
                    
                    
                    # Rate of Return
                    if self.elliptical:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_sma, collision_probability, species, maneuver_cost)
                    else:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_alt, collision_probability, species, maneuver_cost)
                else:
                    if self.elliptical:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_sma, collision_probability, species)
                    else:
                        rate_of_return = open_access.fringe_rate_of_return(state_next_alt, collision_probability, species)
                
                if self.elliptical:
                    solver_guess[:, species.species_idx, 0] = solver_guess[:, species.species_idx, 0] - solver_guess[:, species.species_idx, 0] * (rate_of_return - collision_probability)
                else:
                    solver_guess[species.start_slice:species.end_slice] = solver_guess[species.start_slice:species.end_slice] - solver_guess[species.start_slice:species.end_slice] * (rate_of_return - collision_probability)

            # sto±re the rate of return for this species
            # Check if there are any economic parameters that need to change (e.g demand growth of revenue)
            # multi_species.increase_demand()

            open_access = MultiSpeciesOpenAccessSolver(self.MOCAT, solver_guess, environment_for_solver, "linear", lam, multi_species, years, time_idx, fringe_start_slice, fringe_end_slice, static_maneuver_prices=self.static_maneuver_prices)

            # Solve for equilibrium launch rates
            launch_rate = open_access.solver()

            # Update the initial conditions for the next period
            lam = insert_launches_into_lam(lam, launch_rate, multi_species, self.elliptical)

            elapsed_time = time.time() - start_time
            print(f'Time taken for period {time_idx}: {elapsed_time:.2f} seconds')

            # Update the current environment
            if self.elliptical:
                current_environment = state_next_sma
            else:
                current_environment = state_next_alt

            # # ---- Process Economics ---- # #
            trial_total_revenue = float(open_access._last_total_revenue)
            trial_revenue_by_shell = open_access._last_tax_revenue.tolist()
            trial_species_data = species_data
            trial_launch_rate_by_species = {}
            for sp in multi_species.species:
                trial_launch_rate_by_species[sp.name] = launch_rate[sp.start_slice:sp.end_slice].tolist()
            trial_lam = lam.copy()

            # Welfare: sum of consumer surplus across all endogenous species 
            # (both S and Su contribute per the updated paper), plus any leftover
            # policy revenue that wasn't spent on ADR this period.
            # Uses pre-ADR population plus new launches as the per-period stock.
            consumer_surplus = 0.0
            for sp in multi_species.species:
                if sp.name not in WELFARE_ACTIVE_SPECIES:
                    continue
                sp_pop = np.sum(trial_environment_for_solver[sp.start_slice:sp.end_slice])
                sp_new = np.sum(launch_rate[sp.start_slice:sp.end_slice])
                consumer_surplus += 0.5 * sp.econ_params.coef * (sp_pop + sp_new) ** 2
            welfare = consumer_surplus + trial_leftover_tax_revenue

        # --- NEW CALCULATION: Probability Adjusted OUF ---
            su_species = next((s for s in multi_species.species if s.name == 'Su'), None)
            
            adjusted_ouf_fee = []
            if su_species:
                base_ouf = getattr(su_species.econ_params, 'ouf', 0.0)
                
                su_cp = open_access._last_collision_probability.get('Su', np.zeros(self.MOCAT.scenario_properties.n_shells))

                adjusted_ouf_fee = (base_ouf * su_cp).tolist()
            else:
                adjusted_ouf_fee = np.zeros(self.MOCAT.scenario_properties.n_shells).tolist()
                
            # Save the results that will be used for plotting later
            current_trial_results[cs] = {
                "environment": trial_environment_for_solver.copy(),
                "num_removed": trial_num_removed,
                "new_total_revenue": trial_total_revenue,
                "launch_rate": trial_launch_rate_by_species,
                "lam": trial_lam,
                "removal_dict": removal_dict,
                "species_data": trial_species_data,
                "adjusted_ouf_fee": adjusted_ouf_fee,
                "welfare": welfare,
                "simulation_data": {
                    "ror": rate_of_return,
                    "collision_probability": collision_probability,
                    "launch_rate" : trial_launch_rate_by_species, 
                    "collision_probability_all_species": open_access._last_collision_probability,
                    "umpy": open_access.umpy, 
                    "excess_returns": open_access._last_excess_returns,
                    "non_compliance": open_access._last_non_compliance, 
                    "maneuvers": open_access._last_maneuvers,
                    "maneuver_cost": open_access._last_maneuver_cost,
                    "rate_of_return": open_access._last_rate_of_return,
                    "revenue_total": trial_total_revenue,
                    "revenue_by_shell": trial_revenue_by_shell,
                    "adjusted_ouf_fee": adjusted_ouf_fee,
                    "welfare": welfare,
                    "bond_revenue": open_access.bond_revenue,
                }
            }
            # Track the best shell as we go
            if cs == 1:
                best_welfare_so_far = welfare
                opt_shell = cs
            elif welfare > best_welfare_so_far:
                best_welfare_so_far = welfare
                opt_shell = cs

            # save shell and their welfare in a tuple list of format [(shell_num, welfare),...]
            shell_welfare.append((cs, welfare))

        return current_trial_results, opt_shell, removal_dict, shell_welfare

    def run_optimizer_loop(self, scenario_name, simulation_name, MOCAT_config, params):
        """
            Run the solver for each year using the greedy optimization loop
        """
        simulation_results = {}
        opt_path = {}
        tf, years, current_environment, multi_species, species_data, econ_calculator, shells, lam, fringe_start_slice, fringe_end_slice = OptimizeADR.solve_year_zero(self, MOCAT_config=MOCAT_config, scenario_name=scenario_name, simulation_name=simulation_name, grid_search=False)
        
        # run loop for each year
        for time_idx in tf:
            optimization_trial_results, opt_shell, removal_dict, shell_welfare = OptimizeADR.optimize_adr_loop(self, years=years, time_idx=time_idx, multi_species=multi_species, species_data=species_data, econ_calculator=econ_calculator, current_environment=current_environment, lam=lam, shells=shells, fringe_start_slice=fringe_start_slice, fringe_end_slice=fringe_end_slice)
            
            if opt_shell is not None:
                best_trial_results = optimization_trial_results[opt_shell]

                current_environment = best_trial_results['environment']
                num_actually_removed = best_trial_results['num_removed']
                new_tax_revenue = best_trial_results['new_total_revenue']
                lam = best_trial_results['lam']

               # Now we call process_period_economics to finalize the year's state and prepare the funds for the next period.
                # We ignore the welfare returned here because it does not account for new launches (unlike best_trial_results)
                _, _ = econ_calculator.process_period_economics(
                    num_actually_removed,
                    current_environment,
                    (fringe_start_slice, fringe_end_slice),
                    new_tax_revenue
                        )
                
                # Save the results of the best trial
                simulation_results[time_idx] = best_trial_results['simulation_data']
                
                # Retrieve the correct welfare (inclusive of new launches) from the simulation data
                correct_welfare = best_trial_results['simulation_data']['welfare']

                opt_path[str(time_idx)] = {
                    'Shell':int(opt_shell), 
                    'Num_Removed':int(num_actually_removed), 
                    'Welfare':int(correct_welfare), 
                    'Total_UMPY': int(np.sum(best_trial_results['simulation_data']['umpy']))
                }
                
                # opt_path[str(time_idx)] = best_trial_results['removal_dict']
                # Update the species data with the best trial's data
                for i, sp in enumerate(self.MOCAT.scenario_properties.species_names):
                    species_data[sp] = best_trial_results['species_data'][sp]
                welfare = correct_welfare.copy()
            else:
                welfare, _ = econ_calculator.process_period_economics(
                    num_actually_removed = 0,
                    current_environment=self.environment_before_adr,
                    fringe_slices = (fringe_start_slice, fringe_end_slice),
                    new_tax_revenue = 0, # Corrected from new_t_revenue
                )
                pass
            
            # sorting shell_welfare list based on welfare
            shell_welfare.sort(key=lambda x:x[1])
            new_order = []
            for pair in shell_welfare:
                new_order.append(pair[0])
            
            self.adr_params.shell_order = new_order       
            print(f"New ADR Precedence Order: {new_order}")     
            
            # sammie addition: storing the optimizable values and params
            self.welfare_dict[scenario_name] = welfare

        self.adr_dict[scenario_name] = int(np.sum(simulation_results[tf[-1]]['umpy']))

        removal_save_path = f"./Results/{simulation_name}/{scenario_name}/removal_path.json"
        if not os.path.exists(os.path.dirname(removal_save_path)):
            os.makedirs(os.path.dirname(removal_save_path))
        with open(removal_save_path, 'w') as json_file:
            json.dump(opt_path, json_file, indent=4)

        # opt_save_path = f"./Results/{simulation_name}/{scenario_name}/opt_comparison_values.json"
        # if not os.path.exists(os.path.dirname(opt_save_path)):
        #     os.makedirs(os.path.dirname(opt_save_path))
        # with open(opt_save_path, 'w') as json_file:
        #     json.dump(optimization_trial_results, json_file, indent=4)

        if self.grid_search:
                return species_data
        else:
            # Create a dictionary of econ_params for all species
            all_econ_params = {
                species.name: species.econ_params 
                for species in multi_species.species 
                if hasattr(species, 'econ_params') and species.econ_params is not None
            }
            
            PostProcessing(self.MOCAT, scenario_name, simulation_name, species_data, simulation_results, all_econ_params, grid_search=False)
            return species_data

    def get_mocat_from_optimizer(self):
        return self.MOCAT
    