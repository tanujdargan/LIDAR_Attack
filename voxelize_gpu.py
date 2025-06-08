import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from astropy.stats import knuth_bin_width
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

try:
    import cupy as cp  # Optional GPU support
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    cp = None
    HAS_CUPY = False


class Voxelize:
    """Compute voxel features with optional GPU acceleration."""

    def __init__(self, pointcloud: np.ndarray, use_gpu: bool = False):
        self.use_gpu = use_gpu and HAS_CUPY
        if self.use_gpu:
            self.xp = cp
            self._pc = cp.asarray(pointcloud)
        else:
            self.xp = np
            self._pc = np.asarray(pointcloud)

    def _get_cpu_data(self, arr):
        """Helper to get CPU array if using CuPy, otherwise returns the array as is."""
        return cp.asnumpy(arr) if self.use_gpu else arr

    def __basic_stats(self, col: int):
        """Computes mean, min, max, range, and std for a given column."""
        data = self._pc[:, col]
        
        if data.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        mean_val = float(self.xp.mean(data))
        min_val = float(self.xp.min(data))
        max_val = float(self.xp.max(data))
        range_val = float(max_val - min_val)
        std_val = float(self.xp.std(data))
        return mean_val, min_val, max_val, range_val, std_val

    def __entropy_r(self) -> float:
        """
        Computes the entropy of the reflectance channel (column 3)
        using Sturges' formula for binning.
        """
        data = self._get_cpu_data(self._pc[:, 3])
        
        finite_data = data[np.isfinite(data)]
        if len(finite_data) <= 1: 
            return 0.0

        num_bins = int(np.log2(len(finite_data)) + 1)
        if num_bins < 2:
            num_bins = 2 
        
        hist, _ = np.histogram(finite_data, bins=num_bins)
        
        sum_hist = np.sum(hist)
        if sum_hist == 0:
            return 0.0

        probs = hist / sum_hist
        probs = probs[probs > 0]
        
        return float(entropy(probs, base=2)) if len(probs) > 0 else 0.0

    def __kurtosis_r(self) -> float:
        """Computes the kurtosis of the reflectance channel (column 3)."""
        data = self._get_cpu_data(self._pc[:, 3])
        if len(data) < 4:
            return 0.0
        return float(kurtosis(data))

    def __skewness_r(self) -> float:
        """Computes the skewness of the reflectance channel (column 3)."""
        data = self._get_cpu_data(self._pc[:, 3])
        if len(data) < 2:
            return 0.0
        return float(skew(data, axis=0, bias=False))

    def __percentage_outliers_r(self) -> float:
        """
        Computes the percentage of outliers in the reflectance channel (column 3)
        using the Interquartile Range (IQR) method.
        """
        data = self._get_cpu_data(self._pc[:, 3])
        if len(data) < 5:
            return 0.0

        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (data < lower_bound) | (data > upper_bound)
        return float(np.sum(outliers) / len(data))
    
    def __peak_bin_ratio_r(self) -> float:
        """
        Computes the peak bin ratio for the reflectance channel (column 3)
        using Sturges' formula for binning.
        """
        data = self._get_cpu_data(self._pc[:, 3])
        
        finite_data = data[np.isfinite(data)]
        if len(finite_data) <= 1:
            return 0.0
        
        num_bins = int(np.log2(len(finite_data)) + 1)
        if num_bins < 1:
            num_bins = 1
        
        hist, _ = np.histogram(finite_data, bins=num_bins)

        sum_hist = np.sum(hist)
        if sum_hist == 0:
            return 0.0
        return float(np.max(hist) / sum_hist)

    def __reflectance_region_planarity(self) -> float:
        """
        Computes the planarity (flatness) measure for the high reflectance regions
        within the voxel. A value closer to 0 indicates flatter, higher values
        indicate more volumetric/curved.
        """
        points = self._get_cpu_data(self._pc[:, :3]) # Get XYZ coordinates
        reflectance = self._get_cpu_data(self._pc[:, 3])

        # Handle empty voxel or insufficient points
        if len(points) == 0 or len(reflectance) == 0:
            return 0.0

        # 1. Identify high reflectance points using a percentile threshold (efficient)
        # Using 90th percentile as before, can be adjusted
        high_r_threshold = np.percentile(reflectance, 90) 
        high_r_mask = (reflectance > high_r_threshold)
        
        high_r_points = points[high_r_mask]

        # 2. Compute global planarity (PCA-based) for only these high reflectance points
        num_high_r_points = len(high_r_points)
        if num_high_r_points < 3: # Need at least 3 points to define a plane for PCA
            return 0.0 # Cannot compute planarity

        # Center the high reflectance points
        centroid = np.mean(high_r_points, axis=0)
        centered_high_r_points = high_r_points - centroid

        # Compute the covariance matrix
        # rowvar=False means columns are variables (x, y, z), rows are observations (points)
        cov_matrix = np.cov(centered_high_r_points, rowvar=False) 

        # Compute eigenvalues to determine shape
        # eigvalsh is for symmetric matrices (like covariance)
        try:
            eigvals = np.linalg.eigvalsh(cov_matrix)
            # Sort eigenvalues in ascending order (lambda_0 <= lambda_1 <= lambda_2)
            # Smallest eigenvalue (lambda_0) relates to the "thickness"
            # Sum of eigenvalues relates to total variance
            eigvals.sort() 
            
            sum_eigvals = np.sum(eigvals)
            if sum_eigvals > 0:
                # Planarity = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
                # A value close to 0 indicates high planarity (flat)
                # Higher values indicate more volumetric/curved shape
                return float(eigvals[0] / sum_eigvals)
            else:
                return 0.0 # All high_r_points are co-located or only one point
        except np.linalg.LinAlgError:
            # Handle cases where covariance matrix might be singular (e.g., points are collinear)
            return 0.0

    def voxel(self) -> tuple:
        """
        Computes a comprehensive set of voxel features, now including an efficient
        measure of planarity (flatness/curvature) specifically for high reflectance regions.
        """
        # Basic stats for X, Y, Z, and Reflectance
        x_stats = self.__basic_stats(0)
        y_stats = self.__basic_stats(1)
        z_stats = self.__basic_stats(2)
        r_stats = self.__basic_stats(3)

        # Advanced reflectance features
        entropy_r_val = self.__entropy_r()
        kurtosis_r_val = self.__kurtosis_r()
        skewness_r_val = self.__skewness_r()
        peak_bin_ratio_r_val = self.__peak_bin_ratio_r()
        percentage_outliers_r_val = self.__percentage_outliers_r() 

        # Feature specifically for high reflectance region planarity/curvature
        reflectance_region_planarity_val = self.__reflectance_region_planarity()

        return (
            *x_stats, 
            *y_stats, 
            *z_stats, 
            *r_stats, 
            entropy_r_val, 
            kurtosis_r_val, 
            skewness_r_val,
            peak_bin_ratio_r_val, 
            percentage_outliers_r_val,
            reflectance_region_planarity_val # This is the key feature for your use case
        )


def main(in_csv: str, out_csv: str, use_gpu: bool, batch_size: int = 100) -> None:
    """Voxelize LiDAR files from *in_csv* writing results to *out_csv*.

    Processing is done in batches so memory usage stays low even for very
    large datasets.
    """

    reader = pd.read_csv(in_csv, header=0, chunksize=batch_size)
    first = True
    for chunk in reader:
        voxels = []
        for fname in chunk['filename']:
            pc = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
            voxels.append(str(tuple(Voxelize(pc, use_gpu=use_gpu).voxel())))
        chunk['voxel'] = voxels
        chunk.to_csv(out_csv, mode='w' if first else 'a', index=False,
                     header=first)
        first = False
    if use_gpu and not HAS_CUPY:
        warnings.warn('cupy not available - processed on CPU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxelize LiDAR files with optional GPU acceleration')
    parser.add_argument('--input', default='filetracker_poisoned.csv', help='Input CSV listing LiDAR files')
    parser.add_argument('--output', default='filetracker_poisoned_voxelized.csv', help='Output CSV file')
    parser.add_argument('--cpu', action='store_true', help='Force computation on CPU')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of files to process at once')
    args = parser.parse_args()
    main(args.input, args.output, use_gpu=not args.cpu, batch_size=args.batch_size)
