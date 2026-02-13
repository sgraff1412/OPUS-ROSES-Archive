from pyssem.model import Model
import json

class ADRParameters:
    """
    Class to set up ADR Parameters based on the parameter .json file
    """
    def __init__(self, adr_params_json, mocat: Model):
        # Save MOCAT
        self.mocat = mocat
        self.target_species = None
        self.adr_times = None
        self.n_max = None        
        self.remove_method = None
        self.time = None
        self.removals_left = None
        self.shell_order = None
        self.exogenous = 0
        
    def adr_parameter_setup(self, configuration, baseline=False):
        # finding parameters for the scenario within setup .json file
        file = open('./OPUS/configuration/adr_setup.json')
        adr = json.load(file)
        if (not configuration.startswith("Baseline")) and (configuration in adr):
            params = adr[configuration]
            # times in which ADR occurs
            self.adr_times = params["adr_times"]
            # target species for removal
            self.target_species = params["target_species"]
            # removal based on percentage or based on set number
            self.remove_method = params["remove_method"]
            if "p" in self.remove_method:
                self.p_remove = params["p_remove"]
            if "n" in self.remove_method:
                self.n_remove = params["n_remove"]
            # target shell for removal
            self.target_shell = params["target_shell"]
            # number of shells to remove from in a single year (not used)
            self.n_max = params["n_max"]
            # exogenous (=1) or endogenous (=0) ADR
            self.exogenous = params["exogenous"]
            # shell removal order/precedence (temporarily hard coded)
            self.shell_order = [12, 14, 13, 15, 17, 11, 18, 16, 19, 20, 10, 9, 8, 5, 6, 7, 4, 3, 2, 1]
        else:
            print("No ADR implemented. ")
            self.target_species = []
            self.adr_times = []
            self.target_shell = []
            
