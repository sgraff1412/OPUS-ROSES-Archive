import numpy as np
import os

######################
# ADR Implementation #
######################

def implement_adr(state_matrix, MOCAT, adr_params):
    """
        Implement ADR in scenarios that don't use optimization and have a set amount of removals. 
    """
    num_removed = {}
    if (len(adr_params.target_species) == 0) or (adr_params.target_species is None):
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                if sp in adr_params.target_species:  
                    start = i*MOCAT.scenario_properties.n_shells
                    end = (i+1)*MOCAT.scenario_properties.n_shells
                    old = state_matrix[start:end]
                    num = []
                    if "p" in adr_params.remove_method:
                        # idx = np.where(adr_params.target_species == sp)
                        idx = adr_params.target_species.index(sp)
                        p_remove = adr_params.p_remove[idx]
                        # p_remove = adr_params.p_remove
                        for j in adr_params.target_shell:
                            num.append(state_matrix[start:end][j-1]*p_remove)
                            state_matrix[start:end][j-1] *= (1-p_remove)
                        
                    elif "n" in adr_params.remove_method:
                        idx = adr_params.target_species.index(sp)
                        n_remove = adr_params.n_remove[idx]
                        for j in adr_params.target_shell:
                            if adr_params.removals_left < n_remove:
                                n_remove = adr_params.removals_left
                            if n_remove > state_matrix[start:end][j-1]:
                                n = state_matrix[start:end][j-1] - n_remove
                                n_remove += n
                                state_matrix[start:end][j-1] = 0
                            else:
                                state_matrix[start:end][j-1] -= n_remove
                            num.append(n_remove)
                    num_removed[sp] = {"num_removed":int(np.sum(num))}
                else:
                    state_matrix = state_matrix



    return state_matrix, num_removed

def optimize_ADR_removal(state_matrix, MOCAT, adr_params):
    """
        Implement ADR in optimization scenarios. 
    """
    removal_dict = {}
    indicator = 0
    if (len(adr_params.target_species) == 0) or (adr_params.target_species is None):
        state_matrix = state_matrix
    elif adr_params.target_species is not None:
        if adr_params.time in adr_params.adr_times:
            for i, sp in enumerate(MOCAT.scenario_properties.species_names):
                if sp in adr_params.target_species:  
                    n_shells = MOCAT.scenario_properties.n_shells
                    start = i*n_shells
                    end = (i+1)*n_shells
                    old = state_matrix[start:end]
                    num = []

                    if "p" in adr_params.remove_method:
                        # idx = np.where(adr_params.target_species == sp)
                        idx = adr_params.target_species.index(sp)
                        p_remove = adr_params.p_remove[idx]
                        # p_remove = adr_params.p_remove
                        for j in adr_params.target_shell:
                            # num.append(state_matrix[start:end][j-1]*p_remove)
                            state_matrix[start:end][j-1] *= (1-p_remove)
                        
                    elif "n" in adr_params.remove_method:
                        counter = 1
                        indicator = 0
                        for shell in adr_params.target_shell:
                            idx = adr_params.target_species.index(sp)
                            n_remove = adr_params.removals_left
                            ts = shell
                            if n_remove <= 0:
                                print('No ADR due to budget constraints.')
                                removal_dict[str(ts)] = {}
                                removal_dict[str(ts)]['Implemented'] = 0
                                indicator = 1
                            elif n_remove > state_matrix[start:end][ts- 1]:
                                n = state_matrix[start:end][ts-1] - n_remove
                                removal_dict[str(ts)] = {}
                                removal_dict[str(ts)]['Implemented'] = 1
                                removal_dict[str(ts)]['Exhausted'] = 0
                                removal_dict[str(ts)]['amount_removed'] = int(n_remove + n)
                                removal_dict[str(ts)]['Order'] = int(counter)
                                removal_dict[str(ts)]['Removals_Left'] = int(n*(-1))
                                n_remove = n * (-1)
                                state_matrix[start:end][ts-1] = 0
                                while indicator < 1:
                                    for ii in adr_params.shell_order:
                                        if ((ii not in adr_params.target_shell) and (ii <= n_shells)):
                                            if n_remove > state_matrix[start:end][ii-1]:
                                                counter = counter + 1
                                                n = state_matrix[start:end][ii-1] - n_remove
                                                if n > 0:
                                                    removal_dict[str(ii)] = {}
                                                    removal_dict[str(ii)]['Implemented'] = 0
                                                    removal_dict[str(ii)]['Exhausted'] = 0
                                                    removal_dict[str(ii)]['amount_removed'] = int(n_remove)
                                                    removal_dict[str(ii)]['n'] = int(n)
                                                    removal_dict[str(ii)]['counter'] = int(counter)
                                                    removal_dict[str(ii)]['status'] = 'found a problem'
                                                    indicator = 1
                                                removal_dict[str(ii)] = {}
                                                removal_dict[str(ii)]['Implemented'] = 1
                                                removal_dict[str(ii)]['Exhausted'] = 0
                                                removal_dict[str(ii)]['amount_removed'] = int(n_remove + n)
                                                removal_dict[str(ii)]['Order'] = int(counter)
                                                removal_dict[str(ii)]['Removals_Left'] = int(n*(-1))
                                                if n_remove == 0 or n == 0:
                                                    indicator = 1
                                                n_remove = n * (-1)
                                                state_matrix[start:end][ii-1] = 0
                                            else:
                                                counter = counter + 1
                                                state_matrix[start:end][ii-1] -= n_remove
                                                removal_dict[str(ii)] = {}
                                                removal_dict[str(ii)]['Implemented'] = 1
                                                removal_dict[str(ii)]['Exhausted'] = 0 
                                                removal_dict[str(ii)]['amount_removed'] = int(n_remove)
                                                removal_dict[str(ii)]['Order'] = int(counter)
                                                removal_dict[str(ii)]['Removals_Left'] = int(0)
                                                n_remove = 0
                                                indicator = 1
                                        if counter > n_shells:
                                            removal_dict['final'] = {}
                                            removal_dict['final']['Exhausted'] = 1 
                                            removal_dict['final']['Removals Left'] = int(n_remove)
                                            removal_dict['final']['Counter_Status'] = int(counter)
                                            indicator = 1
                            else:
                                state_matrix[start:end][ts-1] -= n_remove
                                shell_num = str(ts)
                                removal_dict[shell_num] = {}
                                removal_dict[shell_num]['Shell'] = shell_num,
                                removal_dict[shell_num]['amount_removed'] = int(n_remove)
                                removal_dict[shell_num]['Order'] = int(counter) 
                                removal_dict[shell_num]['Removals_Left'] = int(0)
                            
                else:
                    state_matrix = state_matrix
                    removal_dict[str(0)] = {}
                    removal_dict[str(0)]['Implemented'] = 0

    return state_matrix, removal_dict


