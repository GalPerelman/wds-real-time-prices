from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class UncertaintyResult:
    """Holds the estimated mean and covariance."""
    mu: np.ndarray            # shape (p,)
    Sigma: np.ndarray         # shape (p, p)
    # Optional context
    vector_size: int
    variables: List[str]      # in the stacking order used
    hours_per_window: int     # usually 24


class SpatioTemporalEllipsoidUncertaintyEstimator:
    """
    Estimate mean (mu) and covariance (Sigma) of a stacked daily vector:
        y = [var1(h=0..23), var2(h=0..23), ..., varK(h=0..23)]  in R^{24*K}

    Data requirements
    -----------------
    - `df` must be hourly, indexed by pandas.DatetimeIndex (timezone-aware or naive is fine),
      with numeric columns corresponding to variables (e.g., consumers and electricity_price).
    - Rows represent timestamps; columns are variables.
    - The estimator forms DAILY windows; by default it drops days with incomplete 24 hours.

    Seasonality handling
    --------------------
    If `seasonal_demean=True`, we compute per-variable, per-hour-of-day means over the
    available days (baseline), subtract them, and estimate covariance on residuals.
    We still keep the baseline so you can reconstruct a full mean vector if desired.

    Stacking order
    --------------
    Variables are stacked by variable, then hour:
        [var1@h0, var1@h1, ..., var1@h23, var2@h0, ..., varK@h23]
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        hours_per_window: int = 24,
        drop_incomplete_windows: bool = True,
        seasonal_demean: bool = True,
    ):
        self.variables_in = variables
        self.hours_per_window = hours_per_window
        self.drop_incomplete_windows = drop_incomplete_windows
        self.seasonal_demean = seasonal_demean

        # Learned during fit
        self.variables_: List[str] = []
        self.baseline_: Optional[pd.DataFrame] = None  # (H x K) DataFrame
        self.mu_raw_: Optional[np.ndarray] = None
        self.Sigma_raw_: Optional[np.ndarray] = None
        self.mu_resid_: Optional[np.ndarray] = None
        self.Sigma_resid_: Optional[np.ndarray] = None

        # Internal artifacts to support re-estimation for arbitrary horizons
        self._index_map_: Dict[int, Tuple[str, int]] = {}
        self._cube_: Optional[np.ndarray] = None        # shape (D, K, H)
        self._days_: Optional[List[pd.Timestamp]] = None
        self._H_: Optional[int] = None                  # convenience alias for hours_per_window
        self._K_: Optional[int] = None                  # number of variables

    # ------------------------- Public API -------------------------

    def fit(self, df: pd.DataFrame) -> "SpatioTemporalEllipsoidUncertaintyEstimator":
        """Fit on an hourly DataFrame with a DatetimeIndex and numeric columns."""
        df = self._validate_and_prepare(df)
        vars_ = self.variables_in or [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not vars_:
            raise ValueError("No numeric variables found. Provide `variables=` or ensure numeric columns exist.")
        self.variables_ = vars_

        # Daily windows
        H = self.hours_per_window
        day_groups = df.groupby(df.index.normalize())
        complete = []
        for day, g in day_groups:
            if len(g) == H:
                complete.append((day, g.sort_index()))
        if not complete:
            raise ValueError("No complete daily windows found.")

        self._days_ = [d for d, _ in complete]
        cube = np.stack([self._day_to_array(g[self.variables_]) for _, g in complete], axis=0)  # (D,K,H)
        self._cube_ = cube
        self._K_ = cube.shape[1]
        self._H_ = H

        # Seasonality baseline (K x H)
        if self.seasonal_demean:
            baseline_KH = cube.mean(axis=0)  # (K,H)
            self.baseline_ = pd.DataFrame(
                baseline_KH.T,  # (H,K)
                index=pd.RangeIndex(H, name="hour"),
                columns=self.variables_,
            )
            resid_cube = cube - baseline_KH[None, :, :]
        else:
            self.baseline_ = None
            resid_cube = cube

        # Vectorize (D, K*H)
        Y_raw = self._vec_stack(cube)
        Y_resid = self._vec_stack(resid_cube)
        # Sample stats
        self.mu_raw_ = Y_raw.mean(axis=0)
        self.mu_resid_ = Y_resid.mean(axis=0)

        self.Sigma_raw_ = self._make_pd(np.cov(Y_raw, rowvar=False), method="eigs")
        self.Sigma_resid_ = self._make_pd(np.cov(Y_resid, rowvar=False), method="eigs")

        # Index map
        self._index_map_ = self._build_index_map()

        return self

    def get_uncertainty_set(
            self,
            use_residuals: bool = True,
            start_hour: int = 0,
            horizon: Optional[int] = None,
            recompute: bool = False,
    ) -> UncertaintyResult:
        """
        Return an ellipsoidal set for the specified start_hour and horizon.
        - If horizon is None: return the full fitted horizon (default H hours).
        - If horizon == H: rotate to start_hour.
        - If horizon < H: extract_horizon(horizon, start_hour, ...).
        """
        if horizon is None or horizon == self._H_:
            return self.rotate_to_start_hour(start_hour, use_residuals=use_residuals)
        else:
            return self.extract_horizon(
                horizon=horizon,
                start_hour=start_hour,
                use_residuals=use_residuals,
                recompute=recompute,
            )

    def rotate_to_start_hour(
        self,
        start_hour: int,
        use_residuals: bool = True,
    ) -> UncertaintyResult:
        """Rotate hours inside each variable-block so hour=0 becomes `start_hour`."""
        self._require_fit()
        H = self._H_
        K = self._K_
        s = start_hour % H
        mu, Sigma = self._pick_muSigma(use_residuals)

        # permutation indices for a single block of length H
        block_idx = list(range(H))
        rot = block_idx[s:] + block_idx[:s]  # [s..H-1, 0..s-1]

        # build global permutation across K blocks
        perm = []
        for k in range(K):
            base = k * H
            perm.extend([base + r for r in rot])

        mu_rot = mu[perm]
        Sigma_rot = Sigma[np.ix_(perm, perm)]

        return UncertaintyResult(
            mu=mu_rot,
            Sigma=Sigma_rot,
            vector_size=mu_rot.shape[0],
            variables=self.variables_.copy(),
            hours_per_window=H,
        )

    def extract_horizon(
        self,
        horizon: int,
        start_hour: int = 0,
        use_residuals: bool = True,
        recompute: bool = False,
    ) -> UncertaintyResult:
        """
        Produce an ellipsoid for a T-hour horizon starting at `start_hour`.

        - recompute=False: rotate then SELECT first T hours per variable.
        - recompute=True: re-estimate mean/cov from T-length windows starting at `start_hour`
                          (wrap across days if needed). Better when T << H.
        """
        self._require_fit()
        H = self._H_
        K = self._K_
        T = int(horizon)
        if not (1 <= T <= H):
            raise ValueError(f"horizon must be in [1, {H}]")

        if not recompute:
            # Rotate then select
            base = self.rotate_to_start_hour(start_hour, use_residuals=use_residuals)
            # selection indices: keep first T entries from each block
            sel = []
            for k in range(K):
                block_base = k * H
                sel.extend(range(block_base, block_base + T))
            mu_T = base.mu[sel]
            Sigma_T = base.Sigma[np.ix_(sel, sel)]
            return UncertaintyResult(
                mu=mu_T,
                Sigma=Sigma_T,
                vector_size=mu_T.shape[0],
                variables=self.variables_.copy(),
                hours_per_window=T,
            )

        # ---- recompute path: build samples (M, K, T) then estimate ----
        cube = self._cube_  # (D,K,H)
        assert cube is not None
        s = start_hour % H

        # Build windows across days with wrap handling
        # For each day i:
        #   if s+T <= H: take cube[i,:,s:s+T]
        #   else: concat cube[i,:,s:H] with cube[i+1,:,0:(s+T-H)] (needs next day)
        segments = []
        D = cube.shape[0]
        wrap = s + T - H
        if T <= H - s:
            # no wrap; use all D days
            for i in range(D):
                seg = cube[i, :, s:s+T]  # (K,T)
                segments.append(seg)
        else:
            # wrap; need pairs (i, i+1)
            for i in range(D - 1):  # last day can't wrap to next
                left = cube[i, :, s:H]              # (K, H-s)
                right = cube[i + 1, :, 0:wrap]      # (K, wrap)
                seg = np.concatenate([left, right], axis=1)  # (K,T)
                segments.append(seg)

        if len(segments) < 2:
            raise ValueError(
                "Not enough consecutive days to build recomputed windows for the requested start_hour and horizon."
            )

        sample_cube = np.stack(segments, axis=0)  # (M,K,T)

        # Seasonality demeaning for recompute path
        if self.seasonal_demean and self.baseline_ is not None:
            baseline_KH = self.baseline_.to_numpy().T  # (K,H)
            if T <= H - s:
                base_slice = baseline_KH[:, s:s+T]     # (K,T)
                resid_cube = sample_cube - base_slice[None, :, :]
            else:
                left = baseline_KH[:, s:H]
                right = baseline_KH[:, 0:wrap]
                base_slice = np.concatenate([left, right], axis=1)  # (K,T)
                resid_cube = sample_cube - base_slice[None, :, :]
        else:
            resid_cube = sample_cube

        # Vectorize and estimate
        Y = resid_cube.reshape(resid_cube.shape[0], self._K_ * T)  # (M, K*T)
        mu_T = Y.mean(axis=0)
        # Sigma_T = np.cov(Y, rowvar=False)
        Sigma_T = self._cov_regularized(Y)

        return UncertaintyResult(
            mu=mu_T,
            Sigma=Sigma_T,
            vector_size=mu_T.shape[0],
            variables=self.variables_.copy(),
            hours_per_window=T,
        )

    def baseline_matrix(self) -> Optional[pd.DataFrame]:
        return None if self.baseline_ is None else self.baseline_.copy()

    def index_to_var_hour(self, idx: int) -> Tuple[str, int]:
        if not self._index_map_:
            raise RuntimeError("Estimator not fit.")
        return self._index_map_[idx]

    def var_hour_to_index(self, var: str, hour: int) -> int:
        if var not in self.variables_:
            raise KeyError(f"Unknown variable '{var}'. Known: {self.variables_}")
        H = self.hours_per_window
        if not (0 <= hour < H):
            raise ValueError(f"hour must be in [0, {H-1}]")
        return self.variables_.index(var) * H + hour

    # ------------------------- Internal helpers -------------------------
    def _pick_muSigma(self, use_residuals: bool) -> Tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        if use_residuals:
            return self.mu_resid_, self.Sigma_resid_
        else:
            return self.mu_raw_, self.Sigma_raw_

    def _require_fit(self):
        if self._cube_ is None or self.mu_raw_ is None or self.Sigma_raw_ is None:
            raise RuntimeError("Estimator not fit.")

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas.DatetimeIndex.")
        if df.index.freq is None:
            try:
                df = df.asfreq("h")
            except Exception:
                pass
        return df.sort_index()

    def _day_to_array(self, gvars: pd.DataFrame) -> np.ndarray:
        H = self.hours_per_window
        if len(gvars) != H:
            raise ValueError("Internal: day group not of expected length.")
        return gvars.to_numpy(dtype=float).reshape(H, -1).T  # (K,H)

    def _vec_stack(self, cube: np.ndarray) -> np.ndarray:
        D, K, H = cube.shape
        return cube.reshape(D, K * H)

    def _build_index_map(self) -> Dict[int, Tuple[str, int]]:
        K = len(self.variables_)
        H = self.hours_per_window
        return {k * H + h: (self.variables_[k], h) for k in range(K) for h in range(H)}

    def _ensure_symmetric(self, A: np.ndarray) -> np.ndarray:
        """Force exact symmetry to avoid numerical asymmetry."""
        return 0.5 * (A + A.T)

    def _make_pd(
            self,
            Sigma: np.ndarray,
            *,
            method: str = "shrinkage",  # "shrinkage" | "eigs"
            eps_scale: float = 1e-8,  # jitter scale (relative)
            min_eig_floor: float = 1e-10  # eigenvalue floor (absolute-ish after scaling)
    ) -> np.ndarray:
        """
        Return a symmetric PD matrix close to Sigma.

        Scientific justification:
        - method="shrinkage": Ledoit–Wolf/OAS shrinkage covariance (well-conditioned PD estimator).
        - method="eigs": nearest-PSD/PD by eigenvalue flooring + jitter (projection approach).
        """
        Sigma = self._ensure_symmetric(np.asarray(Sigma, dtype=float))

        p = Sigma.shape[0]
        if Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be square.")

        # Fast path: if already PD, return as-is
        try:
            np.linalg.cholesky(Sigma)
            return Sigma
        except np.linalg.LinAlgError:
            pass

        if method == "shrinkage":
            # Try sklearn shrinkage estimators if available
            try:
                from sklearn.covariance import LedoitWolf, OAS  # type: ignore
            except Exception:
                # Fall back if sklearn isn't installed
                method = "eigs"
            else:
                # We need samples to estimate shrinkage. If you only have Sigma here,
                # fall back to eigs. (Alternative: store Y in fit; see note below.)
                method = "eigs"

        if method == "eigs":
            # Eigenvalue flooring + jitter
            w, V = np.linalg.eigh(Sigma)  # Sigma symmetric
            # Scale-aware floor: use average variance as reference
            avg_var = float(np.mean(np.clip(w, 0.0, None)))  # nonnegative part
            if avg_var <= 0.0 or not np.isfinite(avg_var):
                avg_var = 1.0

            floor = max(min_eig_floor * avg_var, 0.0)
            w_clipped = np.maximum(w, floor)
            Sigma_pd = (V * w_clipped) @ V.T
            Sigma_pd = self._ensure_symmetric(Sigma_pd)

            # Add a tiny jitter to guarantee strict PD (helps when many eigenvalues are exactly floor)
            jitter = eps_scale * avg_var
            Sigma_pd = Sigma_pd + jitter * np.eye(p)

            # sanity check
            np.linalg.cholesky(Sigma_pd)
            return Sigma_pd

        raise ValueError(f"Unknown method='{method}'")

    def _cov_regularized(self, Y: np.ndarray, *, method: str = "ledoitwolf") -> np.ndarray:
        """
        Estimate a well-conditioned PD covariance from samples Y (n_samples, n_features).

        method:
          - "ledoitwolf": Ledoit–Wolf shrinkage (recommended default)
          - "oas": Oracle Approximating Shrinkage (often good when samples are very small)
          - "sample+jitter": sample covariance + diagonal loading
        """
        Y = np.asarray(Y, dtype=float)
        n, p = Y.shape

        if method in {"ledoitwolf", "oas"}:
            try:
                from sklearn.covariance import LedoitWolf, OAS  # type: ignore
            except Exception:
                method = "sample+jitter"

        if method == "ledoitwolf":
            Sigma = LedoitWolf().fit(Y).covariance_
            return self._ensure_symmetric(Sigma)

        if method == "oas":
            Sigma = OAS().fit(Y).covariance_
            return self._ensure_symmetric(Sigma)

        if method == "sample+jitter":
            Sigma = np.cov(Y, rowvar=False)
            return self._make_pd(Sigma, method="eigs", eps_scale=1e-8)

        raise ValueError(f"Unknown method='{method}'")


class ForecastErrorUncertaintyEstimator(SpatioTemporalEllipsoidUncertaintyEstimator):
    """
    Estimate mu and Sigma of forecast errors, indexed by (variable, hour of day):
        e = [var1(h=0..23), var2(h=0..23), ..., varK(h=0..23)]  in R^{24*K}

    Input is a hindcast error array (see forecast.Hindcast): one 24-h forecast per hourly issue time.
    Each forecast covers every clock hour exactly once, so its errors are re-ordered from lead order
    to hour-of-day order and every issue time gives one sample vector. Sigma is the plain sample
    covariance over all issue times; mu is the mean error (forecast bias).

    Same stacking order and outputs as the parent class, so rotate_to_start_hour, extract_horizon
    (recompute=False) and get_uncertainty_set are reused unchanged. There is a single Sigma, so
    use_residuals has no effect.
    """
    def __init__(self, hours_per_window: int = 24):
        super().__init__(variables=None, hours_per_window=hours_per_window,
                         drop_incomplete_windows=True, seasonal_demean=False)

    def fit(self, errors: np.ndarray, issue_times: pd.DatetimeIndex,
            variables: List[str]) -> "ForecastErrorUncertaintyEstimator":
        """
        Args:
            errors: ndarray (n_issue_times, horizon, n_variables), actual - forecast, lead 0 = issue hour.
            issue_times: DatetimeIndex of the issue times, length n_issue_times.
            variables: Variable names in the order of the last axis of errors.
        """
        N, T, K = errors.shape
        H = self.hours_per_window
        if T != H:
            raise ValueError(f"Hindcast horizon ({T}) must equal hours_per_window ({H}) so that each "
                             f"forecast covers every hour of the day once.")
        self.variables_ = list(variables)

        # Lead order -> hour-of-day order: clock hour h of a forecast issued at hour s is lead (h - s) mod H
        issue_hours = np.asarray(issue_times.hour)
        lead_of_hour = (np.arange(H)[None, :] - issue_hours[:, None]) % H     # (N, H)
        cube = np.take_along_axis(errors, lead_of_hour[:, :, None], axis=1)   # (N, H, K)
        cube = cube.transpose(0, 2, 1)                                         # (N, K, H)

        self._cube_ = cube
        self._K_ = K
        self._H_ = H
        self.baseline_ = None

        Y = self._vec_stack(cube)                                              # (N, K*H)
        self.mu_raw_ = Y.mean(axis=0)
        self.Sigma_raw_ = np.cov(Y, rowvar=False)
        self.mu_resid_ = self.mu_raw_
        self.Sigma_resid_ = self.Sigma_raw_

        self._index_map_ = self._build_index_map()
        return self

    def extract_horizon(self, horizon: int, start_hour: int = 0, use_residuals: bool = True,
                        recompute: bool = False) -> UncertaintyResult:
        """
        Rotate then select the first `horizon` hours (parent behaviour). recompute=True is not supported:
        the parent's recompute path builds windows from consecutive calendar days, not issue times.
        """
        if recompute:
            raise NotImplementedError("recompute=True is not supported for the forecast-error Sigma.")
        return super().extract_horizon(horizon, start_hour=start_hour, use_residuals=use_residuals,
                                       recompute=False)


class Sampler:
    def __init__(self, mu, cov, p, t, n: int, data=None, seed: int = 42):
        self.mu = mu
        self.cov = cov
        self.p = p  # uncertainty dimension
        self.n = n
        self.t = t

        self.rng = np.random.default_rng(seed=seed)

    def draw_sample(self, sampling_method):
        if sampling_method in ['gaussian', 'ellipsoid', 'normal']:
            return self.draw_gaussian_sample()
        if sampling_method in ['uniform']:
            return self.draw_uniform_sample()
        if sampling_method in ['lognormal', 'log_normal', 'log']:
            return self.draw_lognormal_sample()

    def draw_gaussian_sample(self):
        # Identify indices where mu is not zero
        non_zero_indices = np.nonzero(self.mu)[0]
        # Extract the non-zero entries from mu and the corresponding rows and columns from cov
        mu_non_zero = self.mu[non_zero_indices]
        cov_non_zero = self.cov[np.ix_(non_zero_indices, non_zero_indices)]
        # Draw samples from the modified distribution
        samples_non_zero = self.rng.multivariate_normal(mu_non_zero, cov_non_zero, size=self.n).T
        # Create an output array filled with zeros of the original shape
        output_samples = np.zeros((len(self.mu), self.n))
        # Insert the non-zero samples back into the correct positions
        output_samples[non_zero_indices] = samples_non_zero
        return output_samples

    def draw_uniform_sample(self, n_std=3):
        std_full = np.sqrt(np.diag(self.cov))  # shape (p,)
        low = self.mu - n_std * std_full
        high = self.mu + n_std * std_full
        sample = self.rng.uniform(low=low, high=high, size=(self.n, len(low))).T
        return sample

    def draw_lognormal_sample(self):
        """
        Draw samples from a multivariate log-normal distribution.

        Moment-matched to self.mu so that E[sample_i] = mu_i (consistent with the
        forecast), while producing a right-skewed tail that better captures price
        and demand spikes than a Gaussian.

        Log-space parameters are derived analytically from price-space (mu, Sigma):
            Sigma_log[i,j] = log(1 + Sigma[i,j] / (mu_i * mu_j))
            mu_log[i]      = log(mu_i) - Sigma_log[i,i] / 2

        The subtraction of Sigma_log[i,i]/2 in mu_log is the moment-matching
        correction: for X ~ LogNormal(m, s^2), E[X] = exp(m + s^2/2), so setting
        m = log(mu) - s^2/2 ensures E[X] = mu.
        """
        mu_safe = np.maximum(self.mu, 1e-6)
        # Clamp argument to log: when mu is near-zero, cov[i,j] / (mu_i*mu_j) can push
        # 1+x negative, making log undefined. Floor to 1e-8; eigh+eigenvalue flooring
        # then projects the result to the nearest valid PSD matrix.
        sigma_log = np.log(np.maximum(1 + self.cov / np.outer(mu_safe, mu_safe), 1e-8))
        sigma_log = (sigma_log + sigma_log.T) / 2                  # enforce exact symmetry
        w, V = np.linalg.eigh(sigma_log)
        sigma_log = (V * np.maximum(w, 1e-10)) @ V.T               # floor tiny negative eigenvalues
        mu_log = np.log(mu_safe) - np.diag(sigma_log) / 2          # moment-matching correction
        log_samples = self.rng.multivariate_normal(mu_log, sigma_log, size=self.n).T
        return np.exp(log_samples)

    @staticmethod
    def generate_shared_sample(mu, cov, n: int, sampling_method: str = 'normal', seed: int = 42,
                               use_stable_sampling: bool = True) -> np.ndarray:
        """
        Generate a sample that can be shared across SP optimization and Simulation.

        Args:
            mu: Mean vector
            cov: Covariance matrix
            n: Number of scenarios to generate
            sampling_method: 'normal' or 'uniform'
            seed: Random seed for reproducibility
            use_stable_sampling: If True, use eigendecomposition-based sampling for numerical stability.
                                 Recommended for ill-conditioned covariance matrices.

        Returns:
            Sample array of shape (p, n) where p is the dimension of mu
        """
        rng = np.random.default_rng(seed=seed)

        if sampling_method in ['gaussian', 'ellipsoid', 'normal']:
            non_zero_indices = np.nonzero(mu)[0]
            mu_non_zero = mu[non_zero_indices]
            cov_non_zero = cov[np.ix_(non_zero_indices, non_zero_indices)]

            if use_stable_sampling:
                # Eigendecomposition-based sampling for numerical stability
                samples_non_zero = Sampler._sample_multivariate_normal_stable(
                    mu_non_zero, cov_non_zero, n, rng
                )
            else:
                # Standard Cholesky-based sampling (can fail for ill-conditioned cov)
                samples_non_zero = rng.multivariate_normal(mu_non_zero, cov_non_zero, size=n).T

            output_samples = np.zeros((len(mu), n))
            output_samples[non_zero_indices] = samples_non_zero
            return output_samples
        elif sampling_method in ['uniform']:
            std_full = np.sqrt(np.diag(cov))
            low = mu - 3 * std_full
            high = mu + 3 * std_full
            return rng.uniform(low=low, high=high, size=(n, len(low))).T
        elif sampling_method in ['lognormal', 'log_normal', 'log']:
            mu_safe = np.maximum(mu, 1e-6)
            sigma_log = np.log(np.maximum(1 + cov / np.outer(mu_safe, mu_safe), 1e-8))
            sigma_log = (sigma_log + sigma_log.T) / 2
            w, V = np.linalg.eigh(sigma_log)
            sigma_log = (V * np.maximum(w, 1e-10)) @ V.T
            mu_log = np.log(mu_safe) - np.diag(sigma_log) / 2
            log_samples = rng.multivariate_normal(mu_log, sigma_log, size=n).T
            return np.exp(log_samples)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")

    @staticmethod
    def _sample_multivariate_normal_stable(mu, cov, n, rng, min_eigenvalue_ratio=1e-10):
        """
        Sample from multivariate normal using eigendecomposition for numerical stability.

        This method is more stable than Cholesky decomposition for ill-conditioned
        covariance matrices. It preserves the covariance structure better.

        Args:
            mu: Mean vector of shape (p,)
            cov: Covariance matrix of shape (p, p)
            n: Number of samples
            rng: numpy random generator
            min_eigenvalue_ratio: Minimum eigenvalue as ratio of max eigenvalue.
                                  Eigenvalues below this threshold are clamped.

        Returns:
            Samples of shape (p, n)
        """
        # Eigendecomposition: cov = V @ diag(eigenvalues) @ V.T
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Clamp negative/tiny eigenvalues relative to the maximum
        max_eigenvalue = eigenvalues.max()
        min_allowed = max_eigenvalue * min_eigenvalue_ratio
        eigenvalues_clamped = np.maximum(eigenvalues, min_allowed)

        # Generate standard normal samples: z ~ N(0, I)
        z = rng.standard_normal((n, len(mu)))

        # Transform: x = mu + V @ sqrt(Lambda) @ z.T
        # Equivalently: x = mu + (z @ (V * sqrt(Lambda)).T).T
        sqrt_lambda = np.sqrt(eigenvalues_clamped)
        samples = mu + (z @ (eigenvectors * sqrt_lambda).T)

        return samples.T  # Return shape (p, n)

    @staticmethod
    def diagnose_sample(mu, cov, sample):
        """
        Diagnostic function to check if sample statistics match expected values.

        Args:
            mu: Expected mean vector
            cov: Expected covariance matrix
            sample: Sample array of shape (p, n)

        Returns:
            Dictionary with diagnostic information
        """
        sample_mean = sample.mean(axis=1)
        sample_var = sample.var(axis=1)
        expected_var = np.diag(cov)

        # Mean comparison
        mean_diff = sample_mean - mu
        mean_rel_error = np.abs(mean_diff) / (np.abs(mu) + 1e-10)

        # Variance comparison
        var_ratio = sample_var / (expected_var + 1e-10)

        # Covariance condition number
        eigenvalues = np.linalg.eigvalsh(cov)
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-15)

        return {
            'sample_mean_first5': sample_mean[:5],
            'expected_mean_first5': mu[:5],
            'mean_rel_error_max': mean_rel_error.max(),
            'mean_rel_error_mean': mean_rel_error.mean(),
            'sample_var_first5': sample_var[:5],
            'expected_var_first5': expected_var[:5],
            'var_ratio_first5': var_ratio[:5],
            'var_ratio_mean': var_ratio.mean(),
            'var_ratio_min': var_ratio.min(),
            'var_ratio_max': var_ratio.max(),
            'cov_condition_number': condition_number,
            'cov_min_eigenvalue': eigenvalues.min(),
            'cov_max_eigenvalue': eigenvalues.max(),
            'n_negative_eigenvalues': (eigenvalues < 0).sum(),
        }

    def plot_samples(self, sample):
        k = self.p // self.t
        fig, axes = plt.subplots(k, 1, figsize=(10, 6))
        for i in range(k):
            axes[i].plot(sample[self.t * i: self.t * (i + 1), :], 'C0', alpha=0.3)
            axes[i].plot(self.mu[self.t * i: self.t * (i + 1)], 'k')
            axes[i].grid(True)
