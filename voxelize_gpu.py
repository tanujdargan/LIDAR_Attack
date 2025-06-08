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
        n_neighbors = int(min(max(0.05 * len(data), 20), 100))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        preds = lof.fit_predict(data)
        return float(np.sum(preds == -1) / len(data))

    def __peak_bin_ratio_r(self) -> float:
        data = self._cpu(self._pc[:, 3])
        width, bins = knuth_bin_width(data, return_bins=True)
        hist, _ = np.histogram(data, bins=bins)
        return float(np.max(hist) / np.sum(hist))

    def __curvature_reflectance_region_ratio(self) -> float:
        points = self._cpu(self._pc[:, :3])
        nbr = NearestNeighbors(n_neighbors=122, algorithm="kd_tree").fit(points)
        _, indices = nbr.kneighbors(points)
        curvatures = []
        for i in range(len(points)):
            neighbors = points[indices[i]]
            centroid = np.mean(neighbors, axis=0)
            cov = np.cov((neighbors - centroid).T)
            eigvals = np.linalg.eigvalsh(cov)
            if np.sum(eigvals) > 0:
                curvature = eigvals[0] / np.sum(eigvals)
            else:
                curvature = 0
            curvatures.append(curvature)
        curvatures = np.array(curvatures)

        reflectance = self._cpu(self._pc[:, 3]).reshape(-1, 1)
        reflectance += np.random.normal(0, 1e-5, size=reflectance.shape)
        lof = LocalOutlierFactor(n_neighbors=122, contamination='auto')
        preds = lof.fit_predict(reflectance)
        high_r = np.array(preds == -1)
        high_r_curv = curvatures[high_r]
        return float(np.mean(high_r_curv) / np.mean(curvatures))

    def voxel(self) -> tuple:
        return (
            self.__mean(0), self.__min(0), self.__max(0), self.__range(0), self.__std(0),
            self.__mean(1), self.__min(1), self.__max(1), self.__range(1), self.__std(1),
            self.__mean(2), self.__min(2), self.__max(2), self.__range(2), self.__std(2),
            self.__mean(3), self.__min(3), self.__max(3), self.__range(3), self.__std(3),
            self.__entropy_r(), self.__kurtosis_r(), self.__skewness_r(), self.__peak_bin_ratio_r(),
            self.__curvature_reflectance_region_ratio(),
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
