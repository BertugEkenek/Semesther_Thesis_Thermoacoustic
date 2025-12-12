import numpy as np
class Configuration:
    def __init__(self, config_type):
        self.name = config_type
        self.data_path=None
        self.lsq_method = "trf"        # or "lm", once cond_M_scaled is reasonable
        self.mu_reg_lambda = 0.1      # regularization strength
        self.mu_scaling_mode = "global"   # or "perR"
        self.mu_col_rel_thresh = 1e-3
        self.mu_reg_lambda = 0.05
        self.mu_cpl_logmag_min = -2.0
        self.mu_cpl_logmag_max =  0.2
        self.mu_cpl_phase_max  = np.pi
        self.mu_R_poly_degree = 1
        self.mu_two_stage = True
        self.mu12_prior = 1.0 + 0.0j
        self.mu_cpl_logmag_min = -2.0
        self.mu_cpl_logmag_max =  0.2




        if config_type == "Rijke_tube_1":
            
            self.gamma = 1.4

            self.K = 632700  # Pa m^2
            # Natural angular frequencies (rad/s)
            self.w = [796.4*1j, 2276.8*1j] # w1, w2
            # w R pairings
            self.w_R_table = {
                -1.0: 0 + 2276.00j,
                -0.95: -13.8969 + 2257.43j,
                -0.9:  -28.5283 + 2258.01j,
                -0.85: -43.8909 + 2259.06j,
                -0.8:  -60.167 + 2260.67j,
                -0.75: -77.4647 + 2252.95j,
                -0.7:  -95.9114 + 2266.04j
            }

            self.Z = -1/5.55*1j
            # Mode coupling constants
            self.nu = [0.095, 1.165, 0.334, 0.334] # nu11, nu22, nu12, nu21 
            # Alpha coefficients (m/Pa)
            self.alpha = [8.0 / self.K, 0.5 / self.K] # alpha1, alpha2

            self.Lambda = [-1.0, -1.0] # Lambda1, Lambda2

        elif config_type == "Rijke_tube_2":

            self.gamma = 1.4

            self.K = 632700  # Pa m^2

            self.w = [1219.6*1j, 2887.8*1j] # w1, w2
            
            # w R pairings
            self.w_R_table = {
                -1.0: 0 + 2887.80j,
                -0.95: -11.315 + 2870.92j,
                -0.9:  -23.0613 + 2870.86j,
                -0.85: -35.4738 + 2870.75j,
                -0.8:  -48.6297 + 2870.57j,
                -0.75: -62.6205 + 2870.32j,
                -0.7:  -77.5557 + 2699.99j
            }

            self.Z = -1/0.861*1j

            self.nu = [1.582, 0.067, 0.326,0.326] # nu11, nu22, nu12, nu21 

            self.alpha = [1.0 / self.K, 5.6 / self.K] # alpha1, alpha2

            self.Lambda = [-1.0, 1.0] # Lambda1, Lambda2

        elif config_type == "BRS":

            self.gamma = 1.4

            self.K = 667.9  # Pa m^2

            self.w = [329.7*1j, 1455.4*1j] # w1, w2
            
            # w R pairings
            self.w_R_table = {
                -1.0: 0 + 1455.40j,
                -0.95: -22.4122 + 1444.38j,
                -0.9:  -44.9766 + 1444.32j,
                -0.85: -68.8357 + 1444.21j,
                -0.8:  -94.1493 + 1444.06j,
                -0.75: -121.109 + 1443.84j,
                -0.7:  -149.947 + 1443.54j
            }

            self.Z = -1/47.394*1j

            self.nu = [1.679, 257.215, -20.780, -20.780] # nu11, nu22, nu12, nu21 

            self.alpha = [0.28 / self.K, 0.0014 / self.K] # alpha1, alpha2

            self.Lambda = [-1.0, 1.0] # Lambda1, Lambda2