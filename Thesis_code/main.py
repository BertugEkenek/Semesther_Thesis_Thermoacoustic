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
    config_name = "Rijke_tube_2"            # Alternatives: "Rijke_tube_1", "Rijke_tube_2", "BRS"
    config = Configuration(config_name)
    flame_model_choice = "Padé"             # "Padé" or "Taylor"
    mu_order = "Second"                     # "First" or "Second"
    Galerkin = "Second"                     # "First" or "Second"

    correction = True
    enforce_symmetry = True
    comparison = True

    # --- Multi-branch configuration ---
    fit_branches = [1, 2]             # Later: [1, 2] to use two branches

    tau_plot = 0.004
    tau_train_list = [0.004, 0.007]   # List of tau values to merge
    
    data_paths_map = {}

    save_mu = False
    use_saved_mu = False

    save_solution = False
    use_txt_solutions = False

    
    order = 12
    R_value = -0.7

    # -----------------------------------------------------------
    # 2. Configuration object
    # -----------------------------------------------------------
    config.lsq_method = "trf"          # or "lm"
    config.mu_modelIII_lambda = 0
    config.mu_init_lambda = 0
    config.mu_one_target_lambda = 0
    config.mu_continuation_lambda = 0
    
    # needed for save_solution
    n_values = np.linspace(0.001, 4.0, 11)

    window = 3000
    show_tax = True
    show_fig = True
    save_fig = True

    nprandomsigma = 0.7

    n_last = 4
    number_of_n = 11
    config.mu_bake_rank_one = True
    config.mu_hard_constraint = False

    for t in tau_train_list:
        t_ms = int(t * 1000)
        path_b1 = f"./data/Mu_training_data/{config.name}/{t_ms}ms/tax_{config.name}_first_branch_up_to_n={n_last}_with_number_of_n={number_of_n}_tau={t_ms}ms.mat"
        path_b2 = f"./data/Mu_training_data/{config.name}/{t_ms}ms/tax_{config.name}_second_branch_up_to_n={n_last}_with_number_of_n={number_of_n}_tau={t_ms}ms.mat"
        data_paths_map[t] = {'branch1': path_b1, 'branch2': path_b2}

    branch1_data_path = f"./data/Mu_training_data/{config.name}/{int(tau_plot*1000)}ms/tax_{config.name}_first_branch_up_to_n={n_last}_with_number_of_n={number_of_n}_tau={int(tau_plot*1000)}ms.mat"
    branch2_data_path = f"./data/Mu_training_data/{config.name}/{int(tau_plot*1000)}ms/tax_{config.name}_second_branch_up_to_n={n_last}_with_number_of_n={number_of_n}_tau={int(tau_plot*1000)}ms.mat"          # Path to second branch .mat (when num_acoustic_branches == 2)
    txt_solution_path = "./Results/Solutions/Case_with_noise"


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
        data_paths_map=data_paths_map,     # multi-τ
        branch1_data_path=branch1_data_path,
        branch2_data_path=branch2_data_path,
        txt_solution_path=txt_solution_path,
        use_txt_solutions=use_txt_solutions,
        fit_branches=fit_branches,
        enforce_symmetry=enforce_symmetry,
    )

    mu_pipeline.load_all_data(tau_train_list)
    mu_pipeline.prepare()


    # -----------------------------------------------------------
    # 5. Run top-level workflow
    # -----------------------------------------------------------
    workflow = SolveEigenWorkflow(config, mu_pipeline, logger)
    filename = (
        f"./Results/Plots/{config.name}/{int(tau_plot*1000)}ms/eigenvalues_of_{config.name}_{Galerkin}_order_"
        f"{flame_model_choice}_order_{order}_tau_{int(tau_plot*1000)}ms_R_{int(R_value*100)}.pdf"
    )

    workflow.run(
        correction=correction,
        order=order,
        mu_order=mu_order,
        F_model=F_model,                  
        tau_plot=tau_plot,
        tau_train_list=tau_train_list,
        window=window,
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
        use_txt_solutions=use_txt_solutions,
        enforce_symmetry=enforce_symmetry,
        nprandomsigma=nprandomsigma,
        comparison=comparison
    )
if __name__ == "__main__":
    main()
