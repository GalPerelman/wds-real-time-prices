import math

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.stats import norm

import graphs
import uncertainty_utils
from network import LPNetwork

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class BaseOptModel:
    def __init__(self, network, t, forecast, uset_estimator=None, solver_params=None, solver_display=False, milp=True,
                 final_vol_constraint=True, **kwargs):
        self.network = network
        self.t = t
        self.forecast = forecast  # should be a pandas dataframe, first column prices, rest demands
        self.uset_estimator = uset_estimator  # pre-fitted SpatioTemporalEllipsoidUncertaintyEstimator
        self.solver_params = solver_params
        self.solver_display = solver_display
        self.milp = milp
        self.final_vol_constraint = final_vol_constraint

        # for matrices dimensions
        self.n_fsp, self.n_vsp, self.n_tanks = self.network.n_fsp, self.network.n_vsp, self.network.n_tanks

        # total number of variables per time step
        self.n_var = self.n_fsp + self.n_vsp

        # inequality constraints are saved in dicts {name: mat1, name: mat2..], [name: b1, name:b2..]
        self.ineq_mat, self.ineq_rhs = {}, {}

        # the mass balance mat is saved as a property to support tank simulations
        self.tanks_balance_mat = None

        # For uncertain models
        # Ellipsoid uncertainty set
        self.mu = None
        self.sigma = None
        self.delta = None
        self.p = None  # dimension of uncertainty set

        # Optimization placeholders
        self.formulation_start = None
        self.x = None
        self.y = None
        self.problem = None
        self.omega_param = None
        self.constraints = []

        # empty solution placeholders
        self.x_fsp_value = None
        self.x_vsp_value = None
        self.x_value = None

        # Set the nominal electricity prices and demands for the current solution
        self.apply_forecast()

    def apply_forecast(self):
        self.network.demands = self.forecast.iloc[:, 1: self.network.n_tanks + 1].copy()
        self.network.tariffs = self.forecast.iloc[:, :1].copy()

    def get_bool_vector(self):
        if not self.milp:
            return False
        else:
            bool_vector = ()

            # For each time step, mark the first n_fsp elements as boolean
            for i in range(self.t):
                start_idx = i * (self.n_fsp + self.n_vsp)
                end_idx = start_idx + self.n_fsp
                for _ in range(start_idx, end_idx):
                    bool_vector = bool_vector + (_,)

            bool_vector = (np.array(bool_vector, dtype=int),)
            return bool_vector

    def build_problem(self):
        # build the nominal problem
        self.build_mass_balance()
        self.one_comb_only()

    def build_mass_balance(self):
        """
        Construct the bus power balance matrix for the case of equality constraints
        variables order (matrix columns), for t in range(self.t):
        [fsp 0, fsp 1 ... fsp N | vsp 0, vsp 1 ... vsp N]

        order of constraints in the matrix:
        tank 0 upper bound t=0
        tank 0 upper bound t=1
        ...
        tank 0 upper bound t=T
        tank 1 upper bound t=0
        ...
        tank N upper bound t=T
        tank 0 lower bound t=0
        tank 0 lower bound t=1
        ...
        tank N lower bound t=T

        """
        mat = np.zeros((self.n_tanks * self.t * 2, self.n_var * self.t))
        rhs = np.zeros(self.n_tanks * self.t * 2)
        for tank_idx, (tank_id, tank_data) in enumerate(self.network.tanks.iterrows()):

            # upper bound constraints
            tank_demand = self.network.get_tank_demand(self.t, tank_id)
            for t in range(self.t):
                r = tank_idx * self.t + t
                for fsp_idx, fsp_data in self.network.fsp.iterrows():
                    cols = [self.n_var * _ + fsp_idx - 1 for _ in range(t + 1)]
                    if fsp_data['in'] == tank_id:
                        mat[r, cols] = fsp_data['flow']
                    if fsp_data['out'] == tank_id:
                        mat[r, cols] = -fsp_data['flow']

                for vsp_idx, vsp_data in self.network.vsp.iterrows():
                    if vsp_data['in'] == tank_id:
                        cols = [self.n_var * _ + self.n_fsp + vsp_idx for _ in range(t + 1)]
                        mat[r, cols] = 1

            tanks_ub_rhs = np.tile(-tank_data['init_vol'] + tank_data['max_vol'], self.t)
            tanks_ub_rhs = tanks_ub_rhs + tank_demand.values[:self.t].cumsum()
            rhs[tank_idx * self.t: tank_idx * self.t + self.t] = tanks_ub_rhs

            # lower bound constraints
            for t in range(self.t):
                r = self.n_tanks * self.t + tank_idx * self.t + t
                for fsp_idx, fsp_data in self.network.fsp.iterrows():
                    cols = [self.n_var * _ + fsp_idx - 1 for _ in range(t + 1)]
                    if fsp_data['in'] == tank_id:
                        mat[r, cols] = -fsp_data['flow']
                    if fsp_data['out'] == tank_id:
                        mat[r, cols] = fsp_data['flow']

                for vsp_idx, vsp_data in self.network.vsp.iterrows():
                    if vsp_data['in'] == tank_id:
                        cols = [self.n_var * _ + self.n_fsp + vsp_idx for _ in range(t + 1)]
                        mat[r, cols] = -1

            tanks_lb_rhs = np.tile(tank_data['init_vol'] - tank_data['min_vol'], self.t)
            tanks_lb_rhs = tanks_lb_rhs - tank_demand.values[:self.t].cumsum()
            if self.final_vol_constraint:
                tanks_lb_rhs[-1] = tank_data['init_vol'] - tank_data['final_vol'] - tank_demand.values[:self.t].sum()
            rhs[self.n_tanks * self.t + tank_idx * self.t:
                self.n_tanks * self.t + tank_idx * self.t + self.t] = tanks_lb_rhs

        self.ineq_mat["mass_balance"] = mat
        self.ineq_rhs["mass_balance"] = rhs
        self.tanks_balance_mat = mat
        return mat, rhs

    def get_x_bounds(self):
        single_t_lb = np.hstack([np.tile(0, self.n_fsp),  # lower bound for combs duration
                                 self.network.vsp['min_flow'].values,  # desalination lower bound
                                 ])

        single_t_ub = np.hstack([np.tile(1, self.n_fsp),  # upper bound for combs duration
                                 self.network.vsp['max_flow'].values,  # desalination upper bound
                                 ])

        lb = np.tile(single_t_lb, self.t)
        ub = np.tile(single_t_ub, self.t)
        return lb, ub

    def one_comb_only(self):
        mat = np.zeros((self.network.n_stations * self.t, self.n_var * self.t))
        rhs = np.ones(self.network.n_stations * self.t)
        row = 0
        for t in range(self.t):
            start_col = self.n_var * t

            for i, station in enumerate(self.network.fsp["name"].unique()):
                # num combs for station
                k = len(self.network.fsp.loc[self.network.fsp["name"] == station])

                # idx of first comb for station
                i = self.network.fsp.loc[self.network.fsp["name"] == station].iloc[0].name
                w = self.network.fsp.index.get_loc(i)
                mat[row, start_col + w: start_col + w + k] = 1
                row += 1

        self.ineq_mat["one_comb"] = mat
        self.ineq_rhs["one_comb"] = rhs

    def combine_all_constraints(self):
        a = np.vstack([mat for mat in self.ineq_mat.values()])
        b = np.hstack([rhs for rhs in self.ineq_rhs.values()])
        return a, b

    def objective(self):
        """
        Objective function is to minimize the cost of pumping and desalination.
        The cost is calculated as the sum of the product of power and tariff for each time step.
        """
        power_blocks = np.tile(np.hstack([self.network.fsp['power'].values.reshape(1, -1),
                                          np.zeros((1, self.n_vsp))]), self.t)
        power = power_blocks.ravel()
        tariff_blocks = np.array([np.r_[np.repeat(val, self.n_fsp), np.zeros(self.n_vsp)]
                                  for val in self.network.tariffs['tariff'].values[:self.t]])

        tariff = tariff_blocks.ravel()
        return power, tariff

    def export(self, export_path):
        lb, ub = self.get_x_bounds()
        a, b = self.combine_all_constraints()
        a = np.vstack([a, -np.eye(a.shape[1]), np.eye(a.shape[1])])
        b = np.hstack([b, -lb, ub])
        pd.DataFrame(np.hstack([a, b.reshape(-1, 1)])).to_csv(export_path, index=False)

    def construct_nominal_matrix(self):
        lb, ub = self.get_x_bounds()
        a, b = self.combine_all_constraints()

        a = np.vstack([a, -np.eye(a.shape[1]), np.eye(a.shape[1])])
        b = np.hstack([b, -lb, ub])
        power, tariff = self.objective()

        c = (tariff * power)
        return a, b, c

    def formulate(self, **kwargs):
        self.build_problem()  # build again with every formulation call - ensure updated prices and demands
        a, b, c = self.construct_nominal_matrix()
        self.x = cp.Variable(self.n_var * self.t, boolean=self.get_bool_vector())
        self.problem = cp.Problem(cp.Minimize(c @ self.x), [a @ self.x <= b])

    def solve(self, seed=0):
        """
        Solve the optimization problem.

        Args:
            seed: Random seed for Gurobi solver to ensure reproducibility.
                  MILP solvers use randomness in heuristics and branch-and-bound.
                  Setting a fixed seed ensures identical results across runs.
        """
        # Default solver parameters for reproducibility
        solver_opts = {
            'Seed': seed,           # Fix random seed for reproducibility
            'Threads': 1,           # Single thread for deterministic behavior
        }
        # Merge with user-provided solver params if any
        if self.solver_params:
            solver_opts.update(self.solver_params)

        self.problem.solve(solver=cp.GUROBI, reoptimize=True, verbose=self.solver_display, **solver_opts)
        self.solve_time = self.problem.solver_stats.solve_time
        self.solver_mip_gap = (
            self.problem.solver_stats.extra_stats.MIPGap if self.milp else float('nan')
        )
        if self.solver_display:
            print(f"Status: {self.problem.status} | Optimal value: {self.problem.value} | Model: {type(self).__name__}")
        if self.problem.status in cp.settings.SOLUTION_PRESENT:
            self.extract_solution()
        return self.problem.status, self.problem.value

    def extract_solution(self):
        self.x_fsp_value = self.x.value.reshape(self.t, self.n_fsp + self.n_vsp)[:, :self.n_fsp]
        self.x_vsp_value = self.x.value.reshape(self.t, self.n_fsp + self.n_vsp)[:, self.n_fsp:]
        self.x_value = self.x.value

    def get_nominal(self):
        tariff = self.network.tariffs["tariff"].values[:self.t]
        demands = self.network.demands.values[:self.t, :].flatten(order='F')
        return np.hstack([tariff, demands])

    def preprocess_data_for_uncertainty_sets(self):
        tariff = self.network.tariffs["tariff"]
        demands = self.network.demands
        tariff.index = pd.to_datetime(tariff.index, utc=True)
        demands.index = pd.to_datetime(demands.index, utc=True)

        common_index = tariff.index.intersection(demands.index)

        tariff = tariff.loc[common_index].sort_index()
        demands = demands.loc[common_index].sort_index()

        elec_demand_data = pd.concat([tariff, demands], axis=1)
        elec_demand_data.index = pd.to_datetime(elec_demand_data.index, dayfirst=True)
        return elec_demand_data

    def get_ellipsoid_uncertainty_set(self, start_hour, plot=False):
        """
        Extract mu and Sigma from pre-fitted uset_estimator for the given start_hour.
        Used by RO and sampling-based methods (SP).
        """
        if self.uset_estimator is None:
            raise ValueError("uset_estimator must be provided and pre-fitted")

        u = self.uset_estimator.get_uncertainty_set(horizon=self.t, start_hour=start_hour, use_residuals=False)
        self.mu, self.sigma = u.mu, u.Sigma
        self.delta = np.asarray(np.linalg.cholesky(self.sigma).astype(np.float64))
        self.p = int(self.sigma.shape[0])

        if plot:
            plt.imshow(sigma[:self.t, :self.t], cmap='viridis')
            plt.figure()
            plt.imshow(sigma[self.t:, self.t:], cmap='viridis')
            plt.show()


class RO(BaseOptModel):
    """
    Uncertainty set
    ================
    Starting from a cov matrix - Sigma
    We derive a matrix delta such that Sigma = delta @ delta.T
    Now we need to construct a mapping matrix that governs which rows of delta are active per each constraint
    The mapping matrix has dimensions (n_var, n_constraints)

    The cov matrix dimensions are: n_time_steps (electricity prices) + n_time_steps * n_tanks (demands)
    The mapping should be such that first n_time_steps rows are for electricity prices
    """

    def __init__(self, network, t, uncertainty_set, omega=None, gamma=None, k=1.5, **kwargs):
        super().__init__(network, t, **kwargs)
        self.uncertainty_set = uncertainty_set
        self.omega = omega
        self.gamma = gamma
        self.k = k

        self.a = None
        self.n = None
        self.m = None

        self.T = int(self.t)

    def preprocess(self):
        a, b, c = self.construct_nominal_matrix()
        a = np.asarray(a.astype(np.float64))
        b = np.asarray(b.astype(np.float64))
        c = np.asarray(c.astype(np.float64))
        epigraph_obj = np.hstack([[-1.0], c, [0.0]])
        constraints = np.hstack([np.zeros((a.shape[0], 1)), a, -1 * b.reshape(-1, 1)])

        self.a = np.asarray(np.vstack([epigraph_obj, constraints]).astype(np.float64))
        self.n = int(self.a.shape[1])
        self.m = int(self.a.shape[0])

    def get_mapping_matrix(self):
        """
        Generating the matrix that maps uncertain parameters to constraints for both RO models

        Each constraint has its own mapping matrix of size (n_variables, uncertainty_dimension)
        The map denotes which uncertain parameters (col) affect each coefficient (row) in the constraint

        The first row is the objective variable - no coefficients affected by uncertainty
        The last row is the RHS - can be uncertain and is mapped accordingly
        """

        def build_price_matrix():
            mat = np.zeros((self.n, self.p), dtype=np.float64)
            for t in range(self.t):
                start_row = 1 + t * (self.n_fsp + self.n_vsp)
                mat[start_row: start_row + self.n_fsp + self.n_vsp, t] = 1.0

            return mat

        # def build_demand_matrix(tank_idx, t):
        #     mat = np.zeros((self.n, self.p), dtype=np.float64)
        #     mat[-1, self.t + tank_idx * self.t + t] = 1.0
        #     return mat

        def build_demand_matrix(tank_idx, t):
            mat = np.zeros((self.n, self.p), dtype=np.float64)
            # The RHS of the mass balance constraint at time 't' is a function of
            # the cumulative sum of demands up to 't'. Therefore, this constraint
            # is affected by uncertainty in demands from time 0 to t.
            start_uncertainty_idx = self.t + tank_idx * self.t
            end_uncertainty_idx = start_uncertainty_idx + t + 1
            mat[-1, start_uncertainty_idx:end_uncertainty_idx] = 1.0
            return mat

        matrices = [build_price_matrix()]

        # Mass balance constraints appear in two large blocks:
        # first all tank upper bounds, then all tank lower bounds
        # Respect that ordering so each constraint gets the proper demand mapping.
        tank_rows = []
        tank_rows_lower = []
        for tank_idx in range(self.n_tanks):
            for t in range(self.t):
                tank_rows.append(build_demand_matrix(tank_idx, t))
                tank_rows_lower.append(build_demand_matrix(tank_idx, t))

        matrices.extend(tank_rows)
        matrices.extend(tank_rows_lower)

        # one comb only constraints - no uncertainty mapping
        for i in range(self.t * self.network.n_stations):
            mat = np.zeros((self.n, self.p), dtype=np.float64)
            matrices.append(mat)

        # variable bounds constraints - no uncertainty mapping
        for i in range(self.n_var * self.t * 2):
            mat = np.zeros((self.n, self.p), dtype=np.float64)
            matrices.append(mat)

        return matrices

    def formulate_ellipsoid_rc(self, start_hour=None):
        # build again with every formulation call - ensure updated prices and demands
        self.build_problem()
        self.preprocess()
        # generate the uncertainty set with every formulation call - ensure mu and cov are consistent with start hour
        if start_hour is None:
            start_hour = self.forecast.index[0].hour
        self.get_ellipsoid_uncertainty_set(start_hour=start_hour)

        mapping_mats = self.get_mapping_matrix()
        # Precompute multiplication of mapping matrices with delta
        constraints_mapping_matrices = [mat @ self.delta for mat in mapping_mats]

        self.y = cp.Variable(nonneg=True)  # objective epigraph scalar
        self.x = cp.Variable(self.n - 2, nonneg=True, boolean=self.get_bool_vector())  # decision vector
        x_augmented = cp.hstack([self.y, self.x, 1.0])  # [y; x; 1]  (last entry is constant)

        power, tariff = self.objective()  # we need to multiply the pumps power by the uncertain part of the prices
        constraints = []
        for i in range(self.m):
            mapping_i = constraints_mapping_matrices[i]  # n x p (num variables x uncertainty dimension)

            # Constraints with no uncertain parameters reduce to their nominal form.
            # Skipping the SOC avoids building ~840 redundant CVXPY expressions (one_comb_only + x_bounds rows).
            if not np.any(mapping_i):
                constraints.append(self.a[i, :] @ x_augmented <= 0.0)
                continue

            # applying the power on the uncertain tariffs
            # first row - epigraph fictive objective variable
            # last row - represent the RHS column (fictive variable with constant 1)

            # convert the power row into a column for correct broadcasting
            # a[:, np.newaxis] is equivalent to a.reshape(-1, 1) converting a row to a column
            mapping_i[1: -1, :] *= power[:, np.newaxis]

            tau_i = cp.Variable(nonneg=True)  # one SOC slack per uncertain row
            # SOC of constraint i: ||mapping_mat^T @ x_aug||_2 <= t[i]
            constraints.append(cp.norm(mapping_i.T @ x_augmented, 2) <= tau_i)
            # Linear_i: A_full[i] @ x_aug + omega * t[i] <= 0
            constraints.append(self.a[i, :] @ x_augmented + self.omega * tau_i <= 0.0)

        self.problem = cp.Problem(cp.Minimize(self.y), constraints)

    def get_bou_deviations(self, k: float) -> np.ndarray:
        """
        Extract the deviations for Budget of Uncertainty (Bertsimas–Sim):
        radii ρ_j for each primitive uncertain parameter, sorted similarly to self.mu, self.sigma ordering:
        [prices (t), demands (t * n_tanks)].

        :param k:
            The magnitude of the deviation impact
            This is equivalent to the inverse_phi in the SP model, e.g. the multiplier of the std
        """
        if not hasattr(self, "sigma"):
            raise RuntimeError("sigma not set. Call get_ellipsoid_uncertainty_set(...) first.")

        std_full = np.sqrt(np.diag(self.sigma))  # shape (p,)
        rho = k * std_full  # shape (p,)
        return rho

    def formulate_bou_rc(self, start_hour=None):
        """
        Budget of Uncertainty robust counterpart using dual π, θ formulation (Bertsimas–Sim)
        on the epigraph system:
            [ -y + c^T x <= 0 ]
            [  A x - b   <= 0 ]

        Uncertainty:
        - Prices (first t entries of zeta) affect the objective row.
        - Demands (last t * n_tanks entries) affect the mass-balance rows
          through the cumulative-demand mapping in get_mapping_matrix().

        k             : scales std -> ρ_j = k * std_j
        gamma         : a tuple contain the budget of uncertainty as (gamma_obj, gamma_constraints)
                        allowing tho set different budgets for each uncertainty.
                        this is required to avoid extreme costs in the robust solutions
        """
        # build again with every formulation call - ensure updated prices and demands
        self.build_problem()
        self.preprocess()
        # generate the uncertainty set with every formulation call - ensure mu and cov are consistent with start hour
        if start_hour is None:
            start_hour = self.forecast.index[0].hour
        self.get_ellipsoid_uncertainty_set(start_hour=start_hour)

        mapping_mats = self.get_mapping_matrix()    # list of length m, each (n, p)
        rho = self.get_bou_deviations(self.k)       # shape (p,)

        # Decision variables
        self.y = cp.Variable(nonneg=True)  # epigraph scalar
        self.x = cp.Variable(self.n - 2, nonneg=True, boolean=self.get_bool_vector())  # original decision vector
        x_aug = cp.hstack([self.y, self.x, 1.0])  # [y; x; 1]

        # Bertsimas-Sim aux variables
        pi = cp.Variable(self.m, nonneg=True)  # pi_i per row; entries for non-uncertain rows unused

        constraints = []

        # Number of mass-balance constraints (upper only)
        m_mb = self.t * self.n_tanks

        # row 0          : objective
        # rows 1..m_mb   : tank upper bounds
        # rows m_mb+1..2*m_mb : tank lower bounds

        power, tariff = self.objective()  # we need to multiply the pumps power by the uncertain part of the prices
        gamma_obj, gamma = self.gamma
        for i in range(self.m):
            a_i = self.a[i, :]  # 1 x n
            mapping_i = mapping_mats[i]  # n x p

            # Constraints with no uncertain parameters reduce to their nominal form.
            # Skipping theta allocation avoids building ~840 x p redundant CVXPY variables (one_comb_only + x_bounds rows).
            if not np.any(mapping_i):
                constraints.append(a_i @ x_aug <= 0)
                continue

            # applying the power on the uncertain tariffs
            # convert the power row into a column for correct broadcasting
            # a[:, np.newaxis] is equivalent to a.reshape(-1, 1) converting a row to a column
            mapping_i[1: -1, :] *= power[:, np.newaxis]

            # Restrict to active columns of mapping_i (columns with at least one non-zero entry).
            # Inactive cols have no effect on x_aug - theta_ij=0, avoids redundant CVXPY variables.
            active_cols = np.where(np.any(mapping_i != 0, axis=0))[0]  # indices of non-zero columns, len = k
            rho_active = rho[active_cols]                               # shape (k,)
            effect_active = mapping_i[:, active_cols].T @ x_aug        # shape (k,)
            s_i = cp.multiply(rho_active, effect_active)               # shape (k,)

            # Choose Gamma for this row
            if i == 0:
                # objective row: prices only
                g = gamma_obj
            elif 1 <= i <= 2 * m_mb:
                # mass-balance rows: demands only
                g = gamma
            else:
                # other constraints: no BS uncertainty
                g = 0.0

            theta_i = cp.Variable(len(active_cols), nonneg=True)  # theta over active columns only
            # Robust constraint: a_i^T x_aug + Gamma_i * pi_i + sum_j theta_ij <= 0
            constraints.append(a_i @ x_aug + g * pi[i] + cp.sum(theta_i) <= 0)

            # Linking constraints: theta_ij >= s_i[j] - pi_i
            constraints.append(theta_i >= s_i - pi[i])

        self.problem = cp.Problem(cp.Minimize(self.y), constraints)

    def formulate(self, start_hour=None):
        if self.uncertainty_set.lower() == "ellipsoid":
            self.formulate_ellipsoid_rc(start_hour=start_hour)
        elif self.uncertainty_set.lower() == "bou":
            self.formulate_bou_rc(start_hour=start_hour)
        else:
            raise ValueError(f"Unknown uncertainty set type: {self.uncertainty_set}")


class SP(BaseOptModel):
    """
    Stochastic Programming with SAA objective, chance constraints, and optional CVaR risk term.

    Core formulation
    ----------------
    min  E[cost]  +  lambda_cvar * CVaR(cost)
    s.t. Pr(vol_t >= min_vol) >= alpha_t   for all tanks and time steps

    The chance constraints tighten tank bounds by z(alpha_t) * sigma_demand

    The SAA objective offers two options:
    Objective options (controlled by parameters below)
    ---------------------------------------------------
    1. SAA only (default, lambda_cvar=0):
           min  (1/N) sum_i cost_i

    2. SAA + CVaR (lambda_cvar > 0):
           min  E[cost]  +  lambda_cvar * CVaR_{alpha_start}(cost)
       LP reformulation via auxiliary variables eta, s.

    The quantile objective (objective_quantile) is an alternative to the SAA path entirely.

    Parameters
    ----------
    alpha : tuple (alpha_start, alpha_end)
        Confidence levels for chance constraints and CVaR.
        alpha_start: level at t=0; alpha_end: level at t=T-1 (must be <= alpha_start).
        Use alpha_start == alpha_end for constant confidence across the horizon.
    lambda_cvar : float, optional (default 0.0)
        Weight on CVaR tail-risk term. Zero reduces to pure SAA.
    objective_quantile : float in (0,1), optional
        If set, replaces the SAA objective with a quantile-based objective (ignores lambda_cvar).
    sample : np.ndarray, optional
        Pre-generated sample of shape (p, n). When provided (POR experiments), all methods
        share identical scenarios for fair comparison. When None (MPC), fresh samples are
        drawn at each formulation call using the forecast-centered mu and rotated Sigma.
    """
    def __init__(self, network, t, alpha, lambda_cvar=0.0,
                 objective_quantile=None, sampling_method="normal", n=1000, sample=None, **kwargs):
        super().__init__(network, t, **kwargs)

        # Unpack alpha tuple
        self.alpha_start, self.alpha_end = alpha
        self.lambda_cvar = lambda_cvar
        self.objective_quantile = objective_quantile
        self.n = n  # number of samples
        self.sampling_method = sampling_method
        self._shared_sample = sample  # Pre-generated sample (optional)

        # Validation
        if not (0 < self.alpha_end <= self.alpha_start <= 1):
            raise ValueError(f"Require 0 < alpha_end <= alpha_start <= 1, got alpha={alpha}")

        self.sampler = None
        self.sample = None
        self.delta = None
        self.sigma_rhs = None
        self.eta_cvar = None  # CVaR auxiliary variable (VaR threshold)
        self.s_cvar = None    # CVaR exceedance variables, shape (n,)

    def update_sampler(self):
        """
        Update the sampler object with every formulation call.

        Behavior depends on whether a shared sample was provided at initialization:
        - With shared sample (POR experiments): Uses the pre-generated sample to ensure
          all methods are compared against identical scenarios for fair evaluation.
        - Without shared sample (MPC framework): Generates fresh samples using current
          mu and Sigma, which are consistent with the rolling horizon's start hour.
          This is necessary because the uncertainty set rotates as time-of-day changes.
        """
        if self._shared_sample is not None:
            # Use pre-generated shared sample for consistency across experiments
            self.sample = self._shared_sample
            self.n = self.sample.shape[1]  # Update n to match sample size
        else:
            self.sampler = uncertainty_utils.Sampler(self.mu, self.sigma, p=self.sigma.shape[0], t=self.t, n=self.n)
            # sample shape = (p, n) where p is uncertainty dimension - p = n_tanks + 1
            self.sample = self.sampler.draw_sample(sampling_method=self.sampling_method)

    def get_power_block_3d(self):
        power_blocks = np.concatenate([
            self.network.fsp['power'].values,  # shape (n_fsp,)
            np.zeros(self.n_vsp)  # shape (n_vsp,)
        ])  # shape (n_pumps,)

        # Reshape for broadcasting over samples: (t, 1, n_fsp + n_vsp)
        power_blocks = np.tile(power_blocks, (self.t, 1))  # shape: (t, n_pumps)
        power_blocks_3d = power_blocks[:, None, :]
        return power_blocks_3d

    def sample_objective(self):
        # For each time t and sample s, repeat the price over the n_fsp pumps
        # prices_samples[:, :, None] has shape (t, n_samples, 1)
        # last n_vsp entries stay zero
        prices_samples = self.sample[:self.t, :]  # (t, n)

        ##### OLD VERSION - Should be equivalent to the current code #####
        # power_blocks_3d = self.get_power_block_3d()
        #
        # # Build tariff for all samples. convert prices_samples from (t, n_samples) to (t, n_samples, n_fsp + n_vsp)
        # tariff_blocks = np.zeros((self.t, self.n, self.n_fsp + self.n_vsp))

        # # broadcast price over pumps, shape: (t, n_samples, n_fsp)
        # tariff_blocks[:, :, :self.n_fsp] = prices_samples[:, :, None]
        #
        # # Elementwise multiply to get c for each (t, pump, sample)
        # c_blocks = tariff_blocks * power_blocks_3d  # shape: (t, n_samples, n_fsp + n_vsp)
        #
        # # Flatten time + pumps into one variable index. shape: (n_vars, n_samples)
        # # the axis we want as the last dimension must already be the last dimension in the input before reshape
        # c_samples = c_blocks.transpose(0, 2, 1).reshape(self.t * (self.n_fsp + self.n_vsp), self.n)
        ##################################################################

        power_full = np.concatenate([self.network.fsp['power'].values, np.zeros(self.n_vsp)])  # (n_pumps,)
        c_samples = np.repeat(prices_samples, self.n_fsp + self.n_vsp, axis=0) * np.tile(power_full, self.t)[:, None]
        return c_samples

    def quantile_objective(self):
        power_blocks_3d = self.get_power_block_3d()

        prices_samples = self.sample[:self.t, :]  # (t, n)
        tariff_q = np.quantile(prices_samples, self.objective_quantile, axis=1)  # (t,)
        tariff_blocks_q = np.zeros((self.t, 1, self.n_fsp + self.n_vsp))
        tariff_blocks_q[:, 0, :self.n_fsp] = tariff_q[:, None]  # repeat over pumps
        c_blocks_q = tariff_blocks_q * power_blocks_3d  # (t, 1, n_pumps)
        c_quantile = c_blocks_q.reshape(self.t * (self.n_fsp + self.n_vsp))

        return c_quantile

    def formulate(self, start_hour=None):
        self.build_problem()
        if start_hour is None:
            start_hour = self.forecast.index[0].hour
        self.get_ellipsoid_uncertainty_set(start_hour=start_hour)
        # Replace historical seasonal mean with forecast so scenarios are forecast-centered.
        # Variables stacked as [var0@t0..T-1, var1@t0..T-1, ...] matching uset_estimator.variables_.
        if self._shared_sample is None:
            self.mu = np.concatenate([self.forecast[var].values[:self.t]
                                      for var in self.uset_estimator.variables_])

        # update the sampler with every formulation call - ensure consistency with start hour
        self.update_sampler()

        a, b, c_nominal = self.construct_nominal_matrix()
        self.x = cp.Variable(self.n_var * self.t, boolean=self.get_bool_vector())

        ######  Building the stochastic objective  ########
        # The below functions build the objective coefficients:
        # multiply the pumps power by the sampled tariffs
        # repeat the same price for all pumps for every time step
        # c_all_sample is the matrix of objective coefficients for all samples - can be used for expected objective
        # c_quantile is the vector of objective coefficients for the quantile-based objective
        cvar_constraints = []
        if self.objective_quantile is None:
            # SAA objective - based on average of all samples
            c_all_sample = self.sample_objective()
            cost_scenarios = c_all_sample.T @ self.x  # shape (n,): cost per scenario
            obj = (1 / self.n) * cp.sum(cost_scenarios)

            # Optional CVaR term: lambda_cvar * CVaR_{alpha_start}(cost)
            # LP reformulation via Rockafellar-Uryasev: CVaR = eta + 1/((1-alpha)*N) * sum(s)
            self.eta_cvar = cp.Variable()
            self.s_cvar = cp.Variable(self.n, nonneg=True)
            cvar = self.eta_cvar + (1 / ((1 - self.alpha_start) * self.n)) * cp.sum(self.s_cvar)
            obj = obj + self.lambda_cvar * cvar
            cvar_constraints = [self.s_cvar >= cost_scenarios - self.eta_cvar]
        else:
            c_quantile = self.quantile_objective()
            obj = cp.sum(c_quantile.T @ self.x)

        ######  Building the Chance Constraints  ########
        # We assume the demand part of the uncertainty vector is ordered as:
        # [tank0_t0, ..., tank0_t{T}, tank1_t0, ..., tank1_t{T}, ...].
        # For each tank and time t, the RHS involves the cumulative sum of demands
        # from time 0...t, so each row has ones in those positions.

        # Computing the variance and STD for every constraint RHS
        # We want to develop a variance expression for each constraint.
        # RHS (for each tank and time) = sum of demands up to constraint time for that tank = sum_t(d_tank,t)
        # The variance of the RHS is given as:
        # var(RHS) = var(sum_t d_tank,t) = sum_i sum_j cov(d_i, d_j)
        # If we capture this time dependency via a mapping matrix delta, we can write:
        #     RHS_vector = delta @ demands_vector
        # where each row of delta selects and sums the relevant demands for one constraint.
        # Then, for each constraint i:
        #     var_i   = delta_i @ Sigma_demands @ delta_i.T
        #     sigma_i = sqrt(var_i)
        #
        # b_i = b_i_nom + sum_j(delta_i * std_j * inverse_phi(epsilon))
        m = self.t * self.n_tanks  # number of mass balance constraints (only for upper bound, will double later)
        delta = np.zeros((m, self.t * self.n_tanks))
        for tank_idx in range(self.n_tanks):
            for t in range(self.t):
                delta[self.t * tank_idx + t, self.t * tank_idx: tank_idx * self.t + t + 1] = 1.0

        delta_upper = delta  # shape (m, demands_dim)
        delta_lower = delta  # same mapping: var(sum(d)) = var(-sum(d))
        delta_full = np.vstack([delta_upper, delta_lower])  # shape (2m, demands_dim)

        var_rhs = delta_full @ self.sigma[self.t:, self.t:] @ delta_full.T  # remove prices from sigma, shape (2m, 2m)
        sigma_rhs = np.sqrt(np.diag(var_rhs))  # shape (2m,)
        self.delta = delta          # (m, n_tanks*t) cumulative demand mapping
        self.sigma_rhs = sigma_rhs[:m]  # (m,) std of cumulative demand per constraint

        time_idx = np.tile(np.arange(self.t), self.n_tanks * 2)  # shape (2m,)

        # Linear interpolation of confidence level: alpha_t = alpha_start + t * (alpha_end - alpha_start) / (T - 1)
        # When alpha_start == alpha_end, this gives constant confidence (no temporal decay)
        if self.t > 1:
            alpha_t = self.alpha_start + time_idx * (self.alpha_end - self.alpha_start) / (self.t - 1)
        else:
            alpha_t = np.full_like(time_idx, self.alpha_start, dtype=float)

        # quantiles z_i = Phi^{-1}(alpha_t)
        z = norm.ppf(alpha_t)  # shape (2m,)

        # safe RHS for each constraint
        b_safe = b[:m * 2] - z * sigma_rhs  # shape (2m,)
        constraints = cvar_constraints + [a[:m * 2, :] @ self.x <= b_safe, a[m * 2:, :] @ self.x <= b[m * 2:]]

        self.problem = cp.Problem(cp.Minimize(obj), constraints)




class Simulation:
    """
    Monte Carlo simulation for evaluating optimization solutions.

    Parameters
    ----------
    opt_model : BaseOptModel
        Solved optimization model to evaluate.
    sampling_method : str
        Sampling method ('normal', 'uniform').
    n : int
        Number of scenarios to simulate (ignored if sample is provided).
    sample : np.ndarray, optional
        Pre-generated sample array of shape (p, n) to use for evaluation.

        Usage contexts:
        - POR experiments: Pass a shared sample to ensure all methods are evaluated
          against identical scenarios. This guarantees fair comparison.
        - Standalone evaluation: Leave as None to generate fresh samples.
    """

    def __init__(self, opt_model, sampling_method, n=1000, sample=None):
        self.opt_model = opt_model
        self.sampling_method = sampling_method
        self.x_fsp_value = opt_model.x_fsp_value
        self.x_vsp_value = opt_model.x_vsp_value

        self.t = self.opt_model.t
        self.rng = np.random.default_rng(seed=42)

        if self.opt_model.sigma is None:
            start_hour = self.opt_model.forecast.index[0].hour
            self.opt_model.get_ellipsoid_uncertainty_set(start_hour=start_hour)
        self.p = self.opt_model.sigma.shape[0]

        self.tank_volumes = {}
        self.is_violations = pd.DataFrame()
        self.violation_volume = pd.DataFrame()
        self.costs = pd.Series()

        self.mu = self.opt_model.get_nominal()
        self.cov = self.opt_model.sigma

        # Use shared sample if provided, otherwise generate new sample
        if sample is not None:
            self.sample = sample
            self.n = sample.shape[1]
        else:
            self.n = n
            self.sampler = uncertainty_utils.Sampler(mu=self.mu, cov=self.cov, p=self.p, t=self.t, n=self.n)
            self.sample = self.sampler.draw_sample(sampling_method=self.sampling_method)

        self.calculate_tank_volumes()
        self.calculate_cost()

    def calculate_tank_volumes(self, plot=False):
        for tank_idx, (tank_name, tank_data) in enumerate(self.opt_model.network.tanks.iterrows()):
            net_inflow_mat = self.opt_model.tanks_balance_mat[tank_idx * self.t: tank_idx * self.t + self.t, :]
            net_inflow = net_inflow_mat @ self.opt_model.x_value
            init_vol = np.tile(self.opt_model.network.tanks.loc[tank_name, 'init_vol'], self.t).reshape(-1, 1)

            demand_sample = self.sample[self.t + tank_idx * self.t: self.t + tank_idx * self.t + self.t:, :]
            cumm_demand = demand_sample.cumsum(axis=0)
            vol = init_vol + net_inflow.reshape(-1, 1) - cumm_demand
            if plot:
                plt.figure()
                plt.plot(vol, c='C0', alpha=0.5)
                plt.hlines(tank_data['init_vol'], 0, self.t, colors='k', linestyles='--')
                plt.hlines(tank_data['min_vol'], 0, self.t, colors='red')
                plt.hlines(tank_data['max_vol'], 0, self.t, colors='red')

            self.tank_volumes[tank_name] = vol
            tank_ub = np.tile(tank_data['max_vol'], self.t)
            tank_lb = np.tile(tank_data['min_vol'], self.t)
            if self.opt_model.final_vol_constraint:
                tank_lb[-1] = tank_data['final_vol']
            tank_is_violation = np.logical_or(np.any(vol.T - tank_lb < 0, axis=1), np.any(tank_ub - vol.T < 0, axis=1))

            violated_vol = (np.sum(np.maximum(0, tank_lb.reshape(-1, 1) - vol), axis=1)
                            + np.sum(np.maximum(0, vol - tank_ub.reshape(-1, 1)), axis=1))

            self.is_violations[tank_name] = tank_is_violation
            self.violation_volume[tank_name] = violated_vol

    def calculate_cost(self):
        power_blocks = np.concatenate([
            self.opt_model.network.fsp['power'].values,  # shape (n_fsp,)
            np.zeros(self.opt_model.n_vsp)  # shape (n_vsp,)
        ])  # shape (n_pumps,)

        # Reshape for broadcasting over samples: (t, 1, n_fsp + n_vsp)
        power_blocks = np.tile(power_blocks, (self.t, 1))  # shape: (t, n_pumps)
        power_blocks_3d = power_blocks[:, None, :]

        # Build tariff for all samples. convert prices_samples from (t, n_samples) to (t, n_samples, n_fsp + n_vsp)
        tariff_blocks = np.zeros((self.t, self.n, self.opt_model.n_fsp + self.opt_model.n_vsp))

        # For each time t and sample s, repeat the price over the n_fsp pumps
        # prices_samples[:, :, None] has shape (t, n_samples, 1)
        # last n_vsp entries stay zero
        prices_samples = self.sample[:self.t, :]  # (t, n)
        tariff_blocks[:, :, :self.opt_model.n_fsp] = prices_samples[:, :, None]  # broadcast price over pumps
        # Elementwise multiply to get c for each (t, pump, sample)
        c_blocks = tariff_blocks * power_blocks_3d  # shape: (t, n_samples, n_fsp + n_vsp)

        # Flatten time + pumps into one variable index. shape: (n_vars, n_samples)
        c_samples = c_blocks.reshape(self.t * (self.opt_model.n_fsp + self.opt_model.n_vsp), self.n)
        costs = c_samples.T @ self.opt_model.x_value
        self.costs = pd.Series(costs)

    def summarize(self):
        violations_rate = self.is_violations.any(axis=1).sum() / self.n
        print(f"Violations rate: {violations_rate:.2%} | Violated volume: {self.violation_volume.sum().sum():.2f}")
        return violations_rate
