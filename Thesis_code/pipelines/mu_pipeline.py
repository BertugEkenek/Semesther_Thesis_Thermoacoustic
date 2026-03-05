# pipelines/mu_pipeline.py

from __future__ import annotations

import numpy as np

from utils import logger
from dataio.dataloader import load_data, reshape_EV_trajectories, load_txt_solutions
from pipelines.prep_work import prepare_mu
from fitting import linear_fit as linfit
from fitting import second_order_fit as sofit


def _tau_to_key_ms(tau: float | int) -> int:
    """Canonical tau key: integer milliseconds."""
    if isinstance(tau, (int, np.integer)):
        return int(tau)
    return int(round(float(tau) * 1000.0))


def _key_ms_to_tau(key_ms: int) -> float:
    """Key back to seconds."""
    return float(key_ms) / 1000.0


class MUFITPipeline:
    """
    Canonical dataset lives in self.data_store:

      data_store[tau_ms] = {
        "n": n_vec,
        "R": R_vec,

        "EV0_b1": EV0_branch1 or None,
        "EV_traj_b1": EV_trajectories_branch1 or None,
        "min_size_b1": min_size_b1 or None,
        "EV_traj_b1_reshaped": ... or None,
        "EV_traj_b1_stacked": ... or None,

        "EV0_b2": EV0_branch2 or None,
        "EV_traj_b2": EV_trajectories_branch2 or None,
        "min_size_b2": min_size_b2 or None,
        "EV_traj_b2_reshaped": ... or None,
        "EV_traj_b2_stacked": ... or None,
      }

    fit_branches controls what must exist:
      fit_branches=[1]   => branch 1 only
      fit_branches=[2]   => branch 2 only
      fit_branches=[1,2] => both branches

    Second-order fit requires {1,2}.
    """

    def __init__(
        self,
        config: object,
        # single-τ MAT/TXT (optional)
        txt_solution_path: str | None = None,
        use_txt_solutions: bool = False,

        # multi-τ MAT map (optional)
        data_paths_map: dict[float | int, dict[str, str]] | None = None,

        # behavior
        fit_branches: list[int] | None = None,
    ):
        self.config = config
        self.logger = logger

        self.txt_solution_path = txt_solution_path
        self.use_txt_solutions = bool(use_txt_solutions)

        self.data_paths_map = data_paths_map  # keys can be float seconds or int ms

        self.fit_branches = fit_branches if fit_branches is not None else [1]
        self.fit_branches = list(self.fit_branches)
        if not set(self.fit_branches).issubset({1, 2}):
            raise ValueError(f"fit_branches must be subset of [1,2], got {self.fit_branches}")

        # canonical dataset store
        self.data_store: dict[int, dict] = {}

        # canonical grids (shared across taus)
        self.n: np.ndarray | None = None
        self.R: np.ndarray | None = None

    # ==========================================================
    # Loading entry point
    # ==========================================================
    def load_all_data(self, tau_train_list: list[float | int]):
        """
        Loads raw MAT/TXT data for the requested taus into data_store.
        - If data_paths_map is provided: loads all taus found in data_paths_map
          (and validates tau_train_list is a subset of those keys).
        - If data_paths_map is None: does nothing (caller must load manually).
        """
        tau_keys = [_tau_to_key_ms(t) for t in tau_train_list]
        if not tau_keys:
            raise ValueError("tau_train_list must be non-empty.")

        self._load_multi_tau_from_map()
        missing = [k for k in tau_keys if k not in self.data_store]
        if missing:
            loaded = sorted(self.data_store.keys())
            raise KeyError(f"Requested taus (ms) not loaded: {missing}. Loaded: {loaded}")

        # After raw load, reshape per tau/branch
        self._reshape_all()

        self.logger.info(f"Data loading completed. Loaded taus (ms): {sorted(self.data_store.keys())}")

    # ==========================================================
    # Prepare numeric stacked trajectories
    # ==========================================================
    def prepare(self):
        """Converts reshaped object trajectories to numeric complex arrays via prepare_mu."""
        for tau_ms, d in self.data_store.items():
            # Branch 1
            if 1 in self.fit_branches:
                if d.get("EV_traj_b1_reshaped") is None:
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch 1 reshaped missing.")
                d["EV_traj_b1_stacked"] = prepare_mu(d["EV_traj_b1_reshaped"], d["min_size_b1"])
            else:
                d["EV_traj_b1_stacked"] = None

            # Branch 2
            if 2 in self.fit_branches:
                if d.get("EV_traj_b2_reshaped") is None:
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch 2 reshaped missing.")
                d["EV_traj_b2_stacked"] = prepare_mu(d["EV_traj_b2_reshaped"], d["min_size_b2"])
            else:
                d["EV_traj_b2_stacked"] = None

        self.logger.info("Data preparation completed (numeric conversion).")

    # ==========================================================
    # Internal loaders
    # ==========================================================
    def _load_multi_tau_from_map(self):
        first = True

        # normalize map keys to ms
        map_ms: dict[int, dict[str, str]] = {}
        for tau_key, paths in (self.data_paths_map or {}).items():
            map_ms[_tau_to_key_ms(tau_key)] = paths

        for tau_ms, paths in map_ms.items():
            self.logger.info(f"Loading MAT for tau={_key_ms_to_tau(tau_ms):.6f}s (key={tau_ms}ms)")

            d = {
                "n": None,
                "R": None,

                "EV0_b1": None,
                "EV_traj_b1": None,
                "min_size_b1": None,

                "EV0_b2": None,
                "EV_traj_b2": None,
                "min_size_b2": None,
            }

            # Branch 1
            if 1 in self.fit_branches:
                (n1, R1, EV0_1, EV_traj_1, _, _, min1) = load_data(
                    paths["branch1"], show_position=False, show_max_min=False
                )
                d["n"], d["R"] = n1, R1
                d["EV0_b1"], d["EV_traj_b1"], d["min_size_b1"] = EV0_1, EV_traj_1, min1

            # Branch 2
            if 2 in self.fit_branches:
                (n2, R2, EV0_2, EV_traj_2, _, _, min2) = load_data(
                    paths["branch2"], show_position=False, show_max_min=False
                )
                if d["n"] is None:
                    d["n"], d["R"] = n2, R2
                else:
                    if not np.allclose(d["n"], n2) or not np.allclose(d["R"], R2):
                        raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: mismatch between branch 1 and 2 grids.")
                d["EV0_b2"], d["EV_traj_b2"], d["min_size_b2"] = EV0_2, EV_traj_2, min2

            if d["n"] is None or d["R"] is None:
                raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: nothing loaded; check fit_branches and map paths.")

            # global grid consistency across taus
            if first:
                self.n = d["n"]
                self.R = d["R"]
                first = False
            else:
                if not np.allclose(self.n, d["n"]):
                    raise ValueError(f"n grid mismatch at tau={_key_ms_to_tau(tau_ms)}.")
                if not np.allclose(self.R, d["R"]):
                    raise ValueError(f"R grid mismatch at tau={_key_ms_to_tau(tau_ms)}.")

            self.data_store[tau_ms] = d

    # ==========================================================
    # Reshape helper
    # ==========================================================
    def _reshape_all(self):
        for tau_ms, d in self.data_store.items():
            # Branch 1 reshape
            if 1 in self.fit_branches:
                if not np.isfinite(d.get("min_size_b1", np.inf)):
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: min_size_branch1 not finite: {d.get('min_size_b1')}")
                d["EV_traj_b1_reshaped"] = reshape_EV_trajectories(d["EV_traj_b1"], d["min_size_b1"])
                if d["EV_traj_b1_reshaped"] is None:
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch1 reshape returned None.")
            else:
                d["EV_traj_b1_reshaped"] = None

            # Branch 2 reshape
            if 2 in self.fit_branches:
                if not np.isfinite(d.get("min_size_b2", np.inf)):
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: min_size_branch2 not finite: {d.get('min_size_b2')}")
                d["EV_traj_b2_reshaped"] = reshape_EV_trajectories(d["EV_traj_b2"], d["min_size_b2"])
                if d["EV_traj_b2_reshaped"] is None:
                    raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch2 reshape returned None.")
            else:
                d["EV_traj_b2_reshaped"] = None

    # ==========================================================
    # helper
    # ==========================================================
    def get_tau_keys_ms(self) -> list[int]:
        return sorted(self.data_store.keys())

    def ev0_flat(self, branch_id: int, tau: float | int | None = None) -> np.ndarray:
        if not self.data_store:
            raise ValueError("data_store is empty. Call load_all_data()+prepare() first.")

        if tau is None:
            tau_ms = self.get_tau_keys_ms()[0]
        else:
            tau_ms = _tau_to_key_ms(tau)

        d = self.data_store[tau_ms]
        if branch_id == 1:
            return np.hstack(d["EV0_b1"]).ravel()
        if branch_id == 2:
            return np.hstack(d["EV0_b2"]).ravel()
        raise ValueError("branch_id must be 1 or 2")

    def stacked(self, branch_id: int, tau: float | int | None = None) -> np.ndarray:
        if tau is None:
            tau_ms = self.get_tau_keys_ms()[0]
        else:
            tau_ms = _tau_to_key_ms(tau)

        d = self.data_store[tau_ms]
        if branch_id == 1:
            st = d.get("EV_traj_b1_stacked")
            if st is None:
                raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch1 stacked missing.")
            return st
        if branch_id == 2:
            st = d.get("EV_traj_b2_stacked")
            if st is None:
                raise ValueError(f"tau={_key_ms_to_tau(tau_ms)}: branch2 stacked missing.")
            return st
        raise ValueError("branch_id must be 1 or 2")

    # ==========================================================
    # Linear μ-fit
    # ==========================================================
    def find_mu_fit(self, config, tau_train_list: list[float | int], branch_id: int = 1):
        """
        First-order μ-fit, SINGLE-BRANCH (branch_id=1 or 2).
        Stacks regression rows across all taus in tau_train_list.
        """
        if branch_id not in (1, 2):
            raise ValueError("branch_id must be 1 or 2")

        tau_keys = [_tau_to_key_ms(t) for t in tau_train_list]
        if not tau_keys:
            raise ValueError("tau_train_list must be non-empty.")

        missing = [k for k in tau_keys if k not in self.data_store]
        if missing:
            raise KeyError(f"Requested taus (ms) not loaded: {missing}. Loaded: {self.get_tau_keys_ms()}")

        n = self.n
        R = self.R
        num_R = len(R)

        mu_array = np.zeros((num_R, 2), dtype=float)

        for i in range(num_R):
            A_all, b_all = [], []

            for tau_ms in tau_keys:
                d = self.data_store[tau_ms]
                tau = _key_ms_to_tau(tau_ms)

                if branch_id == 1:
                    stacked = d["EV_traj_b1_stacked"]
                    s_ref = np.hstack(d["EV0_b1"]).ravel()[i]
                    cfg = config.get_branch_config(1)
                else:
                    stacked = d["EV_traj_b2_stacked"]
                    s_ref = np.hstack(d["EV0_b2"]).ravel()[i]
                    cfg = config.get_branch_config(2)

                w = stacked[i, 0, :].imag
                sigma = stacked[i, 0, :].real

                A = linfit.build_A(n, tau, w, sigma, s_ref=s_ref)
                b = linfit.build_b(cfg, n, tau, w, sigma, s_ref=s_ref)

                A_all.append(A)
                b_all.append(b)

            A_tot = np.vstack(A_all)
            b_tot = np.concatenate(b_all)

            mu = linfit.regression(A_tot, b_tot, check_condition_number=True, quiet=False)
            mu_array[i, :] = mu

            self.logger.info(
                f"[Linear μ | branch={branch_id}] R[{i}]={R[i]} | μ={mu} | cond(A_tot)={np.linalg.cond(A_tot):.2e}"
            )

        return mu_array, R

    def find_mu_fit_linear_selected_branches(self, config, tau_train_list: list[float | int]):
        results = {}
        for b in self.fit_branches:
            mu_array, R = self.find_mu_fit(config=config, tau_train_list=tau_train_list, branch_id=b)
            results[b] = {"mu": mu_array, "R": R}
        return results

    # ==========================================================
    # Second-order μ-fit
    # ==========================================================
    def find_mu_fit_second_order(
        self,
        config: object,
        tau_train_list: list[float | int],
    ):
        if set(self.fit_branches) != {1, 2}:
            raise ValueError("Second-order μ-fit requires fit_branches=[1,2].")

        # validate strategy presence early (nice error)
        if not hasattr(config, "mu_fit_strategy"):
            raise AttributeError("config.mu_fit_strategy must be set for second-order μ-fit (e.g. 'rank1_sym', 'sym_only', 'none', 'rank1_mag_phasefree').")

        tau_keys = [_tau_to_key_ms(t) for t in tau_train_list]
        if not tau_keys:
            raise ValueError("tau_train_list must be non-empty.")

        missing = [k for k in tau_keys if k not in self.data_store]
        if missing:
            raise KeyError(f"Requested taus (ms) not loaded: {missing}. Loaded: {self.get_tau_keys_ms()}")

        tau_init = min(tau_keys)

        n = self.n
        R = self.R
        num_R = len(R)

        p_list = []
        prev_p = None

        for i in range(num_R):
            # init μ11, μ22 (optional): take from tau_init dataset
            init_mu11 = None
            init_mu22 = None
            try:
                d_init = self.data_store[tau_init]
                st1 = d_init["EV_traj_b1_stacked"]
                st2 = d_init["EV_traj_b2_stacked"]
                ev1 = np.hstack(d_init["EV0_b1"]).ravel()
                ev2 = np.hstack(d_init["EV0_b2"]).ravel()

                def _linear_mu_complex(branch_id, tau_s, w_vec, sig_vec, s_ref):
                    cfg = config.get_branch_config(branch_id)
                    A = linfit.build_A(n, tau_s, w_vec, sig_vec, s_ref)
                    b = linfit.build_b(cfg, n, tau_s, w_vec, sig_vec, s_ref)
                    mu_re_im = linfit.regression(A, b, check_condition_number=True, quiet=True)
                    return mu_re_im[0] + 1j * mu_re_im[1]

                tau_s = _key_ms_to_tau(tau_init)

                w1 = st1[i, 0, :].imag
                sig1 = st1[i, 0, :].real
                init_mu11 = _linear_mu_complex(1, tau_s, w1, sig1, s_ref=ev1[i])

                w2 = st2[i, 0, :].imag
                sig2 = st2[i, 0, :].real
                init_mu22 = _linear_mu_complex(2, tau_s, w2, sig2, s_ref=ev2[i])

            except Exception:
                init_mu11 = None
                init_mu22 = None

            # build blocks across taus
            data_blocks = []
            for tau_ms in tau_keys:
                d = self.data_store[tau_ms]
                tau_s = _key_ms_to_tau(tau_ms)

                ev1 = np.hstack(d["EV0_b1"]).ravel()
                ev2 = np.hstack(d["EV0_b2"]).ravel()

                st1 = d["EV_traj_b1_stacked"]
                st2 = d["EV_traj_b2_stacked"]

                s1 = ev1[i]
                s2 = ev2[i]

                w1 = st1[i, 0, :].imag
                sig1 = st1[i, 0, :].real
                data_blocks.append(
                    dict(tag=f"tau={tau_s:.4f}_b1", tau=tau_s, w=w1, sigma=sig1, s_ref=s1, s_1=s1, s_2=s2)
                )

                w2 = st2[i, 0, :].imag
                sig2 = st2[i, 0, :].real
                data_blocks.append(
                    dict(tag=f"tau={tau_s:.4f}_b2", tau=tau_s, w=w2, sigma=sig2, s_ref=s2, s_1=s1, s_2=s2)
                )

            p_opt, info = sofit.solve_mu_per_R(
                config=config,
                n=n,
                data_blocks=data_blocks,
                quiet=True,
                init_mu11=init_mu11,
                init_mu22=init_mu22,
                prev_p_opt=prev_p,
            )

            prev_p = p_opt
            p_list.append(p_opt)

            self.logger.info(
                f"[Second-order μ] R[{i}]={R[i]:.2e} | "
                f"strategy={getattr(config, 'mu_fit_strategy', '???')} | "
                f"cost={info.get('global_cost', np.nan):.3e} | "
                f"phys_cost={info.get('phys_cost', np.nan):.3e} | "
                f"phys_rms={info.get('phys_rms', np.nan):.3e}"
            )

        mu_array = np.asarray(p_list, dtype=float)

        # Return EV0 for branch1 from anchor tau_init (for plotting consistency)
        EV0_ret = np.hstack(self.data_store[tau_init]["EV0_b1"]).ravel()

        return mu_array, R, EV0_ret