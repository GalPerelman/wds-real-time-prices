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
    def __init__(self, network, t, forecast, database, solver_params=None, solver_display=False, **kwargs):
        self.network = network
        self.t = t
        self.forecast = forecast
        self.database = database
        self.solver_params = solver_params
        self.solver_display = solver_display

        # for matrices dimensions
        self.n_fsp, self.n_vsp, self.n_tanks = self.network.n_fsp, self.network.n_vsp, self.network.n_tanks

        # total number of variables per time step
        self.n_var = self.n_fsp + self.n_vsp

        # inequality constraints are saved in dicts {name: mat1, name: mat2..], [name: b1, name:b2..]
        self.ineq_mat, self.ineq_rhs = {}, {}

        self.apply_forecast()

        self.build_mass_balance()
        self.one_comb_only()

        self.tanks_balance_mat = None  # the mass balance mat is saved as a property to support tank simulations
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

    def apply_forecast(self):
        self.network.demands = self.forecast.iloc[:, 1:].copy()
        self.network.tariffs = self.forecast.iloc[:, :1].copy()

    def get_bool_vector(self):
        bool_vector = np.hstack([np.tile(1, self.n_fsp),
                                 np.tile(0, self.n_vsp),
                                 ])
        bool_vector = [tuple(np.tile(bool_vector, self.t))]
        return bool_vector

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
                        cols = [self.n_var * _ + self.n_fsp + vsp_idx - 1 for _ in range(t + 1)]
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
                        cols = [self.n_var * _ + self.n_fsp + vsp_idx - 1 for _ in range(t + 1)]
                        mat[r, cols] = -1

            tanks_lb_rhs = np.tile(tank_data['init_vol'] - tank_data['min_vol'], self.t)
            tanks_lb_rhs = tanks_lb_rhs - tank_demand.values[:self.t].cumsum()
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

    def formulate(self):
        a, b, c = self.construct_nominal_matrix()
        self.x = cp.Variable(self.n_var * self.t, boolean=False)  # self.get_bool_vector())
        prob = cp.Problem(cp.Minimize(c @ self.x), [a @ self.x <= b])
        prob.solve(solver=cp.ECOS)
        print("Status:", prob.status)
        print("Optimal value", prob.value)

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

        tariff = tariff.loc[common_index]
        demands = demands.loc[common_index]
        elec_demand_data = pd.concat([tariff, demands], axis=1)
        elec_demand_data.index = pd.to_datetime(elec_demand_data.index, dayfirst=True)
        return elec_demand_data

    def get_ellipsoid_uncertainty_set(self, database, plot=False):
        """
        Can be used for RO but also for sampling-based methods where we need mu and Sigma
        """
        # preprocessed_data = self.preprocess_data_for_uncertainty_sets()
        est = uncertainty_utils.SpatioTemporalEllipsoidUncertaintyEstimator(
            seasonal_demean=True,  # remove hour-of-day baseline
            drop_incomplete_windows=True  # require full 24h days
        )
        est.fit(database)
        u = est.get_uncertainty_set(horizon=self.t, start_hour=0, use_residuals=False)
        self.mu, self.sigma = u.mu, u.Sigma

        if plot:
            plt.imshow(sigma[:self.t, :self.t], cmap='viridis')
            plt.figure()
            plt.imshow(sigma[self.t:, self.t:], cmap='viridis')
            plt.show()
        return self.mu, self.sigma


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
    def __init__(self, network, t, omega, **kwargs):
        super().__init__(network, t, **kwargs)
        self.omega = omega

        self.a = np.asarray(self.preprocess().astype(np.float64))

        # ellipsoid uncertainty set
        self.mu = None
        self.sigma = None

        self.get_ellipsoid_uncertainty_set(self.database)
        self.delta = np.linalg.cholesky(self.sigma)
        self.delta = np.asarray(self.delta.astype(np.float64))

        self.T = int(self.t)
        self.n = int(self.a.shape[1])
        self.m = int(self.a.shape[0])
        self.p = int(self.sigma.shape[0])


    def preprocess(self):
        a, b, c = self.construct_nominal_matrix()
        a = np.asarray(a.astype(np.float64))
        b = np.asarray(b.astype(np.float64))
        c = np.asarray(c.astype(np.float64))
        epigraph_obj = np.hstack([[-1.0], c, [0.0]])
        constraints = np.hstack([np.zeros((a.shape[0], 1)), a, -1 * b.reshape(-1, 1)])
        return np.vstack([epigraph_obj, constraints])


    def get_mapping_matrix(self):
        """
        This method generate the matrix that maps uncertain parameters to constraints
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

    def formulate_ellipsoid_rc(self):
        mapping_mats = self.get_mapping_matrix()
        # Precompute multiplication of mapping matrices with delta
        constraints_mapping_matrices = [mat @ self.delta for mat in mapping_mats]

        self.y = cp.Variable(nonneg=True)  # objective epigraph scalar
        self.x = cp.Variable(self.n - 2, nonneg=True)  # decision vector
        x_augmented = cp.hstack([self.y, self.x, 1.0])  # [y; x; 1]  (last entry is constant)

        tau = cp.Variable(self.m, nonneg=True)  # one SOC slack per row

        constraints = []
        for i in range(self.m):
            # SOC of constraint i: ||mapping_mat^T @ x_aug||_2 <= t[i]
            constraints.append(cp.norm(constraints_mapping_matrices[i].T @ x_augmented, 2) <= tau[i])
            # Linear_i: A_full[i] @ x_aug + omega * t[i] <= 0
            constraints.append(self.a[i, :] @ x_augmented + self.omega * tau[i] <= 0.0)

        prob = cp.Problem(cp.Minimize(self.y), constraints)
        prob.solve(solver=cp.MOSEK)
        print("Status:", prob.status)
        print("Optimal value", prob.value)
        self.extract_solution()
        return prob.status, prob.value


class SP(BaseOptModel):
    """ Stochastic Optimization with Sample Average Approximation (SAA) and Chance Constraints """
    def __init__(self, network, t, epsilon, alpha, objective_quantile=None, sampling_method="normal", n=100, **kwargs):
        super().__init__(network, t, **kwargs)

        self.epsilon = epsilon  # allowed violation rate
        self.alpha = alpha  # decay rate for time-varying epsilons
        self.objective_quantile = objective_quantile
        self.n = n  # number of samples
        self.sampling_method = sampling_method

        self.mu = None
        self.sigma = None
        self.get_ellipsoid_uncertainty_set(self.database)

        self.sampler = uncertainty_utils.Sampler(self.mu, self.sigma, p=self.n_tanks + 1, n=self.n,
                                                 sampling_method=self.sampling_method)
        self.sample = self.sampler.draw_sample()  # shape (p, n) where p is uncertainty dimension - p = n_tanks + 1

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
        power_blocks_3d = self.get_power_block_3d()

        # Build tariff for all samples. convert prices_samples from (t, n_samples) to (t, n_samples, n_fsp + n_vsp)
        tariff_blocks = np.zeros((self.t, self.n, self.n_fsp + self.n_vsp))

        # For each time t and sample s, repeat the price over the n_fsp pumps
        # prices_samples[:, :, None] has shape (t, n_samples, 1)
        # last n_vsp entries stay zero
        prices_samples = self.sample[:self.t, :]  # (t, n)
        tariff_blocks[:, :, :self.n_fsp] = prices_samples[:, :,
                                           None]  # broadcast price over pumps, shape: (t, n_samples, n_fsp)

        # Elementwise multiply to get c for each (t, pump, sample)
        c_blocks = tariff_blocks * power_blocks_3d  # shape: (t, n_samples, n_fsp + n_vsp)

        # Flatten time + pumps into one variable index. shape: (n_vars, n_samples)
        c_samples = c_blocks.reshape(self.t * (self.n_fsp + self.n_vsp), self.n)

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

    def formulate(self):
        a, b, c_nominal = self.construct_nominal_matrix()
        self.x = cp.Variable(self.n_var * self.t, boolean=False)  # self.get_bool_vector())

        ######  Building the stochstic objective  ########
        # The below functions build the objective coefficients:
        # multiply the pumps power by the sampled tariffs
        # repeat the same price for all pumps for every time step
        # c_all_sample is the matrix of objective coefficients for all samples - can be used for expected objective
        # c_quantile is the vector of objective coefficients for the quantile-based objective
        if self.objective_quantile is None:
            # SAA objective - based on average of all samples
            c_all_sample = self.sample_objective()
            obj = (1 / self.n) * cp.sum(c_all_sample.T @ self.x)
        else:
            c_quantile = self.quantile_objective()
            obj = cp.sum(c_quantile.T @ self.x)

        ######  Building the Chance Constraints  ########
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

        time_idx = np.tile(np.arange(self.t), self.n_tanks * 2)  # shape (2m,)
        safety_coef = np.minimum(0.2, self.epsilon * (self.alpha ** time_idx))  # shape (2m,)

        # quantiles z_i = phi^{-1}(1 - safety_coef)
        z = norm.ppf(1 - safety_coef)  # shape (2m,)

        # safe RHS for each constraint
        b_safe = b[:m * 2] - z * sigma_rhs  # shape (2m,)
        constraints = [a[:m * 2, :] @ self.x <= b_safe, a[m * 2:, :] @ self.x <= b[m * 2:]]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.ECOS)
        print("Status:", prob.status)
        print("Optimal value", prob.value)
        self.extract_solution()
        return prob.status, prob.value



class Simulation:
    def __init__(self, opt_model, n=1000):

        self.opt_model = opt_model
        self.x_fsp_value = opt_model.x_fsp_value
        self.x_vsp_value = opt_model.x_vsp_value

        self.n = n
        self.t = self.opt_model.t

        self.tank_volumes = {}
        self.violations = pd.DataFrame()

        self.sample = self.draw_sample()
        self.calculate_tank_volumes()

    def draw_sample(self):
        rng = np.random.default_rng(seed=42)
        mu = self.opt_model.get_nominal()

        # Identify indices where mu is not zero
        non_zero_indices = np.nonzero(mu)[0]
        # Extract the non-zero entries from mu and the corresponding rows and columns from cov
        mu_non_zero = mu[non_zero_indices]
        cov_non_zero = self.opt_model.sigma[np.ix_(non_zero_indices, non_zero_indices)]
        # Draw samples from the modified distribution
        samples_non_zero = rng.multivariate_normal(mu_non_zero, cov_non_zero, size=self.n).T
        # Create an output array filled with zeros of the original shape
        output_samples = np.zeros((len(self.opt_model.mu), self.n))
        # Insert the non-zero samples back into the correct positions
        output_samples[non_zero_indices] = samples_non_zero
        return output_samples

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
            tank_lb[-1] = tank_data['final_vol']
            tank_lb, tank_ub = tank_lb, tank_ub
            tank_violations = np.logical_or(np.any(vol.T - tank_lb < 0, axis=1), np.any(tank_ub - vol.T < 0, axis=1))
            self.violations[tank_name] = tank_violations

    def summarize(self):
        violations_rate = self.violations.any(axis=1).sum() / self.n
        print(f"Violations rate: {violations_rate:.2%}")
        return violations_rate
