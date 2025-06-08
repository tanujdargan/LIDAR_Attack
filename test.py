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
    """Compute voxel features with optional GPU acceleration and memory optimization."""

    def __init__(self, pointcloud: np.ndarray, use_gpu: bool = False, max_points: int = 50000):
        self.use_gpu = use_gpu and HAS_CUPY
        self.max_points = max_points
        
        # Subsample if point cloud is too large
        if len(pointcloud) > max_points:
            indices = np.random.choice(len(pointcloud), max_points, replace=False)
            pointcloud = pointcloud[indices]
            print(f"Subsampled point cloud from {len(pointcloud)} to {max_points} points")
        
        if self.use_gpu:
            self.xp = cp
            self._pc = cp.asarray(pointcloud)
        else:
            self.xp = np
            self._pc = np.asarray(pointcloud)

    def _cpu(self, arr):
        return cp.asnumpy(arr) if self.use_gpu else arr

    def __mean(self, col) -> float:
        return float(self.xp.mean(self._pc[:, col]))

    def __std(self, col) -> float:
        return float(self.xp.std(self._pc[:, col]))

    def __min(self, col) -> float:
        return float(self.xp.min(self._pc[:, col]))

    def __max(self, col) -> float:
        return float(self.xp.max(self._pc[:, col]))

    def __range(self, col) -> float:
        col_d = self._pc[:, col]
        return float(self.xp.max(col_d) - self.xp.min(col_d))

    def __entropy_r(self) -> float:
        data = self._cpu(self._pc[:, 3])
        hist, _ = np.histogram(data, bins="auto")
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        return float(entropy(probs, base=2))

    def __kurtosis_r(self) -> float:
        data = self._cpu(self._pc[:, 3])
        return float(kurtosis(data))

    def __skewness_r(self) -> float:
        data = self._cpu(self._pc[:, 3])
        return float(skew(data, axis=0, bias=False))

    def __percentage_outliers_r(self) -> float:
        data = self._cpu(self._pc[:, 3]).reshape(-1, 1)
        n_neighbors = min(int(min(max(0.05 * len(data), 20), 100)), len(data) - 1)
        if n_neighbors < 2:
            return 0.0
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        preds = lof.fit_predict(data)
        return float(np.sum(preds == -1) / len(data))

    def __peak_bin_ratio_r(self) -> float:
        data = self._cpu(self._pc[:, 3])
        width, bins = knuth_bin_width(data, return_bins=True)
        hist, _ = np.histogram(data, bins=bins)
        return float(np.max(hist) / np.sum(hist))

    def __curvature_reflectance_region_ratio_optimized(self) -> float:
        """Memory-optimized version using batch processing and sampling."""
        points = self._cpu(self._pc[:, :3])
        n_points = len(points)
        
        # Further subsample for curvature computation if still too large
        max_curvature_points = 10000
        if n_points > max_curvature_points:
            indices = np.random.choice(n_points, max_curvature_points, replace=False)
            sample_points = points[indices]
            sample_reflectance = self._cpu(self._pc[indices, 3])
        else:
            sample_points = points
            sample_reflectance = self._cpu(self._pc[:, 3])
        
        # Adaptive number of neighbors based on point cloud size
        k_neighbors = min(max(int(0.01 * len(sample_points)), 10), 50)
        
        try:
            nbr = NearestNeighbors(n_neighbors=k_neighbors, algorithm="kd_tree").fit(sample_points)
            _, indices = nbr.kneighbors(sample_points)
        except Exception:
            # Fallback: return a default value if neighbor computation fails
            return 1.0
        
        # Process curvatures in batches to save memory
        batch_size = 1000
        curvatures = []
        
        for start_idx in range(0, len(sample_points), batch_size):
            end_idx = min(start_idx + batch_size, len(sample_points))
            batch_indices = indices[start_idx:end_idx]
            
            batch_curvatures = []
            for i, neighbor_indices in enumerate(batch_indices):
                neighbors = sample_points[neighbor_indices]
                centroid = np.mean(neighbors, axis=0)
                centered = neighbors - centroid
                
                # Use more stable covariance computation
                if len(centered) > 3:
                    try:
                        cov = np.cov(centered.T)
                        eigvals = np.linalg.eigvalsh(cov)
                        eigvals = np.real(eigvals)  # Ensure real values
                        eigvals = eigvals[eigvals > 1e-10]  # Filter near-zero eigenvalues
                        
                        if len(eigvals) > 0 and np.sum(eigvals) > 1e-10:
                            curvature = eigvals[0] / np.sum(eigvals)
                        else:
                            curvature = 0.0
                    except:
                        curvature = 0.0
                else:
                    curvature = 0.0
                
                batch_curvatures.append(curvature)
            
            curvatures.extend(batch_curvatures)
        
        curvatures = np.array(curvatures)
        
        # Simplified outlier detection for reflectance
        reflectance = sample_reflectance.reshape(-1, 1)
        reflectance += np.random.normal(0, 1e-5, size=reflectance.shape)
        
        # Use smaller neighbor count for outlier detection
        outlier_neighbors = min(k_neighbors, 20)
        try:
            lof = LocalOutlierFactor(n_neighbors=outlier_neighbors, contamination='auto')
            preds = lof.fit_predict(reflectance)
            high_r = np.array(preds == -1)
        except:
            # Fallback: use simple statistical outlier detection
            q75, q25 = np.percentile(reflectance.flatten(), [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            high_r = (reflectance.flatten() < lower_bound) | (reflectance.flatten() > upper_bound)
        
        if np.sum(high_r) > 0 and np.mean(curvatures) > 0:
            high_r_curv = curvatures[high_r]
            return float(np.mean(high_r_curv) / np.mean(curvatures))
        else:
            return 1.0

    def __curvature_reflectance_region_ratio_simple(self) -> float:
        """Simplified alternative using statistical measures instead of nearest neighbors."""
        points = self._cpu(self._pc[:, :3])
        reflectance = self._cpu(self._pc[:, 3])
        
        # Simple curvature approximation using local variance
        if len(points) < 10:
            return 1.0
        
        # Use spatial binning instead of k-NN for efficiency
        n_bins = min(50, int(np.sqrt(len(points))))
        
        try:
            # Create 3D histogram for spatial binning
            hist, edges = np.histogramdd(points, bins=n_bins)
            
            # Simple curvature measure based on spatial variance
            curvature_measure = np.var(hist) / (np.mean(hist) + 1e-10)
            
            # Simple outlier detection for reflectance
            q75, q25 = np.percentile(reflectance, [75, 25])
            iqr = q75 - q25
            outlier_threshold = q75 + 1.5 * iqr
            high_reflectance_ratio = np.sum(reflectance > outlier_threshold) / len(reflectance)
            
            # Combine measures
            return float(curvature_measure * high_reflectance_ratio + 1.0)
        
        except:
            return 1.0

    def voxel(self, use_simple_curvature: bool = True) -> tuple:
        """
        Compute voxel features.
        
        Args:
            use_simple_curvature: If True, use simplified curvature computation
                                 to avoid memory issues. If False, use optimized
                                 but more memory-intensive version.
        """
        if use_simple_curvature:
            curvature_feature = self.__curvature_reflectance_region_ratio_simple()
        else:
            curvature_feature = self.__curvature_reflectance_region_ratio_optimized()
        
        return (
            self.__mean(0), self.__min(0), self.__max(0), self.__range(0), self.__std(0),
            self.__mean(1), self.__min(1), self.__max(1), self.__range(1), self.__std(1),
            self.__mean(2), self.__min(2), self.__max(2), self.__range(2), self.__std(2),
            self.__mean(3), self.__min(3), self.__max(3), self.__range(3), self.__std(3),
            self.__entropy_r(), self.__kurtosis_r(), self.__skewness_r(), self.__peak_bin_ratio_r(),
            curvature_feature,
        )


def main(in_csv: str, out_csv: str, use_gpu: bool, batch_size: int = 100, 
         max_points: int = 50000, use_simple_curvature: bool = True) -> None:
    """
    Voxelize LiDAR files from *in_csv* writing results to *out_csv*.

    Processing is done in batches so memory usage stays low even for very
    large datasets.
    
    Args:
        in_csv: Input CSV file path
        out_csv: Output CSV file path  
        use_gpu: Whether to use GPU acceleration
        batch_size: Number of files to process at once
        max_points: Maximum points per cloud (will subsample if exceeded)
        use_simple_curvature: Use simplified curvature computation
    """
    reader = pd.read_csv(in_csv, header=0, chunksize=batch_size)
    first = True
    
    for chunk_idx, chunk in enumerate(reader):
        print(f"Processing chunk {chunk_idx + 1}...")
        voxels = []
        
        for i, fname in enumerate(chunk['filename']):
            try:
                pc = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
                print(f"  Processing {fname}: {len(pc)} points")
                
                voxelizer = Voxelize(pc, use_gpu=use_gpu, max_points=max_points)
                voxel_features = voxelizer.voxel(use_simple_curvature=use_simple_curvature)
                voxels.append(str(tuple(voxel_features)))
                
            except Exception as e:
                print(f"  Error processing {fname}: {e}")
                # Add default values if processing fails
                voxels.append(str(tuple([0.0] * 25)))
        
        chunk['voxel'] = voxels
        chunk.to_csv(out_csv, mode='w' if first else 'a', index=False, header=first)
        first = False
        
    if use_gpu and not HAS_CUPY:
        warnings.warn('cupy not available - processed on CPU')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxelize LiDAR files with optional GPU acceleration')
    parser.add_argument('--input', default='filetracker_poisoned.csv', help='Input CSV listing LiDAR files')
    parser.add_argument('--output', default='filetracker_poisoned_voxelized.csv', help='Output CSV file')
    parser.add_argument('--cpu', action='store_true', help='Force computation on CPU')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of files to process at once')
    parser.add_argument('--max-points', type=int, default=50000, help='Maximum points per cloud (subsample if exceeded)')
    parser.add_argument('--simple-curvature', action='store_true', help='Use simplified curvature computation')
    
    args = parser.parse_args()
    main(args.input, args.output, use_gpu=not args.cpu, batch_size=args.batch_size,
         max_points=args.max_points, use_simple_curvature=args.simple_curvature)