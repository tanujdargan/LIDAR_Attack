# LIDAR_Attack
The code for the adverserial and physical attack on Lidar sensors in MASS.

## Types of Attacks

There are two types of attacks[1]:
- Saturation attacks (Denial of Service)
- Spoofing attacks

## Saturation attacks
Jamming or blinding LiDAR sensors by emitting strong light in same wavelength as the LiDAR sensor.
### Detection Mechanism
This attack can be detected by taking a LiDAR file of type '.bin' and
then we proceed to reduce the points in the file into one tuple
consisting of ~25 features. These are 
- x_mean
- x_max
- x_min
- x_std
- y_mean
- y_std
- y_min
- y_max
- z_mean
- z_min
- z_max
- z_std
- r_max
- r_mean
- r_std
- r_min
- range_x 
- range_y
- range_z
- range_r
- kurtosis_r
- skewness_in_r 
- percentage_outlier_r
- Peak_bin_ratio_r 
- entropy_of_r
- curvature_reflectance_region_ratio

After computing these values, we proceed to compute ML algorithms for the same.



## References
[1] S. B. Jakobsen, K. S. Knudsen, and B. Andersen, “Analysis of Sensor Attacks against Autonomous Vehicles: 8th International Conference on Internet of Things, Big Data and Security,” Proceedings of the 8th International Conference on Internet of Things, Big Data and Security - IoTBDS, vol. 1, pp. 131–139, 2023, doi: 10.5220/0011841800003482.
