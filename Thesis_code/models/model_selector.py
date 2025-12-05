# model_selector.py

from models.characteristic import (
    characteristic_poly_model3,
    characteristic_poly_model3_1mode,
    characteristic_poly_model3_2mode,
)

def select_characteristic_model(correction: bool, mu_order: str, Galerkin: str):
    """
    Returns the appropriate characteristic polynomial function.
    Removes all branching from solve_roots().
    """

    if not correction:
        # Base (uncorrected) model — depends only on Galerkin
        base_registry = {
            "First": characteristic_poly_model3_1mode,
            "Second": characteristic_poly_model3,
        }
        if Galerkin not in base_registry:
            raise ValueError(f"Unknown Galerkin: {Galerkin}")
        return base_registry[Galerkin]

    # Corrected model
    if mu_order == "First":
        if Galerkin != "First":
            raise ValueError(
                "First-order μ with correction requires Galerkin == 'First'"
            )
        return characteristic_poly_model3_1mode

    if mu_order == "Second":
        return characteristic_poly_model3_2mode

    raise ValueError(f"Unknown mu_order: {mu_order}")
