# main.py

import numpy as np
from config.configuration import Configuration
from pipelines.mu_pipeline import MUFITPipeline
from pipelines.orchestrator import SolveEigenWorkflow
from models.flame_models import F_pade, F_taylor
from utils import logger

def main():
    
    # -----------------------------------------------------------
    # 1. User-configurable parameters
    # -----------------------------------------------------------
    config_name = "Rijke_tube_1"            # Alternatives: "Rijke_tube_1", "Rijke_tube_2", "BRS"
    config = Configuration(config_name)
    flame_model_choice = "Padé"             # "Padé" or "Taylor"
    mu_order = "First"                     # "First" or "Second"
    Galerkin = "First"                     # "First" or "Second"

    correction = True
    enforce_symmetry = True
    use_only_acoustic = False


    # --- Multi-branch configuration ---
    num_acoustic_branches = 2            # Set to 2 when using two .mat files
    fit_branches = [1, 2]                # Later: [1, 2] to use two branches
    
    save_mu = False
    use_saved_mu = False

    save_solution = False
    use_txt_solutions = False

    tau = 0.004
    order = 7
    R_value = -1.0
    
    # needed for save_solution
    n_values = np.linspace(0.001, 4.0, 11)

    tolerance = 2000
    show_tax = True
    show_fig = True
    save_fig = True

    nprandomsigma = 0.5


    # -----------------------------------------------------------
    # 2. Configuration object
    # -----------------------------------------------------------
    config.lsq_method = "trf"          # or "lm"
    config.mu_reg_lambda = 5  # try 0.1, 0.2, etc.
    config.mu_two_stage = False
    config.data_path = f"./data/Mu_training_data/{int(tau*1000)}ms/tax_{config.name}_tau_{int(tau*1000)}ms.mat"
    #config.data_path = f"./data/Mu_training_data/{int(tau*1000)}ms/tax_{config.name}.mat"
    config.txt_solution_path = "./Results/Solutions/Reference_case.txt"
    branch2_data_path = f"./data/Mu_training_data/{int(tau*1000)}ms/tax_{config.name}_second_branch_tau_{int(tau*1000)}ms.mat"          # Path to second branch .mat (when num_acoustic_branches == 2)
    #branch2_data_path = None          # Path to second branch .mat (when num_acoustic_branches == 2)


    # -----------------------------------------------------------
    # 3. Select flame model (cached internally)
    # -----------------------------------------------------------
    if flame_model_choice == "Padé":
        F_model = F_pade
    elif flame_model_choice == "Taylor":
        F_model = F_taylor
    else:
        raise ValueError(f"Unknown flame model: {flame_model_choice}")


    # -----------------------------------------------------------
    # 4. Build μ-fit pipeline
    # -----------------------------------------------------------
    mu_pipeline = MUFITPipeline(
        config=config,
        data_path=config.data_path,
        txt_solution_path=config.txt_solution_path,
        order=order,
        use_only_acoustic=use_only_acoustic,
        use_txt_solutions=use_txt_solutions,
        enforce_symmetry=enforce_symmetry,

        # --- NEW multi-branch arguments ---
        num_acoustic_branches=num_acoustic_branches,
        fit_branches=fit_branches,
        branch2_data_path=branch2_data_path,
    )


    mu_pipeline.load_all_data()
    mu_pipeline.prepare()


    # -----------------------------------------------------------
    # 5. Run top-level workflow
    # -----------------------------------------------------------
    workflow = SolveEigenWorkflow(config, mu_pipeline, logger)

    filename = (
        f"eigenvalues_of_{config.name}_"
        f"{flame_model_choice}_order_{order}_tau_{tau}_R_{R_value}.pdf"
    )

    workflow.run(
        correction=correction,
        order=order,
        mu_order=mu_order,
        F_model=F_model,                  
        tau=tau,
        tolerance=tolerance,
        R_value=R_value,
        n_values=n_values,
        Galerkin=Galerkin,
        show_tax=show_tax,
        save_fig=save_fig,
        show_fig=show_fig,
        filename=filename,
        save_mu=save_mu,
        use_saved_mu=use_saved_mu,
        save_solution=save_solution,
        use_only_acoustic=use_only_acoustic,
        use_txt_solutions=use_txt_solutions,
        enforce_symmetry=enforce_symmetry,
        nprandomsigma=nprandomsigma,
    )


if __name__ == "__main__":
    main()
