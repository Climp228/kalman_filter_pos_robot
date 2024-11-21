import numpy as np
import matplotlib.pyplot as plt

# Параметри симуляції
np.random.seed(42)
num_steps = 50
true_position = np.cumsum(np.random.randn(num_steps))  # Істинне положення
measurement_noise = np.random.normal(0, 1, num_steps)  # Шум вимірювань
measurements = true_position + measurement_noise

# Параметри фільтрів
process_variance = 1  # Дисперсія процесу
measurement_variance = 1  # Дисперсія вимірювань


# Комплементарний фільтр
def complementary_filter(measurements, alpha=0.9):
    estimates = []
    estimate = 0
    for measurement in measurements:
        estimate = alpha * (estimate + measurement) + (1 - alpha) * measurement
        estimates.append(estimate)
    return np.array(estimates)


# Частинковий фільтр
def particle_filter(measurements, num_particles=1000):
    particles = np.random.normal(0, 10, num_particles)  # Ініціалізація частинок
    estimates = []
    for measurement in measurements:
        # Розсіювання та оновлення частинок
        particles += np.random.normal(0, process_variance, num_particles)
        weights = np.exp(-0.5 * ((particles - measurement) ** 2) / measurement_variance)
        weights /= np.sum(weights)  # Нормалізація вагів
        estimate = np.sum(particles * weights)  # Оцінка
        estimates.append(estimate)

        # Ресемплінг частинок
        indices = np.random.choice(num_particles, num_particles, p=weights)
        particles = particles[indices]
    return np.array(estimates)


# 1. Стандартний фільтр Калмана
def kalman_filter(measurements, process_variance, measurement_variance):
    n = len(measurements)
    estimates = np.zeros(n)
    error_covariances = np.zeros(n)

    # Ініціалізація
    estimate = 0.0
    error_covariance = 1.0

    for i in range(n):
        # Прогнозування
        estimate = estimate
        error_covariance = error_covariance + process_variance

        # Оновлення
        kalman_gain = error_covariance / (error_covariance + measurement_variance)
        estimate = estimate + kalman_gain * (measurements[i] - estimate)
        error_covariance = (1 - kalman_gain) * error_covariance

        estimates[i] = estimate
        error_covariances[i] = error_covariance

    return estimates, error_covariances


# 2. Розширений фільтр Калмана (EKF)
def extended_kalman_filter(measurements, process_variance, measurement_variance):
    n = len(measurements)
    estimates = np.zeros(n)
    error_covariances = np.zeros(n)

    estimate = 0.0
    error_covariance = 1.0

    for i in range(n):
        # Прогнозування (додамо умовну нелінійність, наприклад, логарифм)
        estimate = np.log(np.abs(estimate) + 1)  # Нелінійна частина
        error_covariance = error_covariance + process_variance

        # Оновлення
        kalman_gain = error_covariance / (error_covariance + measurement_variance)
        estimate = estimate + kalman_gain * (measurements[i] - estimate)
        error_covariance = (1 - kalman_gain) * error_covariance

        estimates[i] = estimate
        error_covariances[i] = error_covariance

    return estimates, error_covariances


# 3. Беззапаховий фільтр Калмана (UKF)
def unscented_kalman_filter(measurements, process_variance, measurement_variance):
    n = len(measurements)
    estimates = np.zeros(n)
    error_covariances = np.zeros(n)

    estimate = 0.0
    error_covariance = 1.0

    alpha = 1e-3
    beta = 2
    kappa = 0

    for i in range(n):
        # Генерація сигма-точок
        sigma_points = [estimate]
        sigma_points.append(estimate + np.sqrt((error_covariance + process_variance) * (alpha ** 2 * (n + kappa))))
        sigma_points.append(estimate - np.sqrt((error_covariance + process_variance) * (alpha ** 2 * (n + kappa))))

        # Прогнозування
        predict_estimate = np.mean(sigma_points)
        predict_error_covariance = np.var(sigma_points) + process_variance

        # Оновлення
        kalman_gain = predict_error_covariance / (predict_error_covariance + measurement_variance)
        estimate = predict_estimate + kalman_gain * (measurements[i] - predict_estimate)
        error_covariance = (1 - kalman_gain) * predict_error_covariance

        estimates[i] = estimate
        error_covariances[i] = error_covariance

    return estimates, error_covariances


# Обчислення оцінок для всіх фільтрів
kalman_estimates, _ = kalman_filter(measurements, process_variance, measurement_variance)
ekf_estimates, _ = extended_kalman_filter(measurements, process_variance, measurement_variance)
ukf_estimates, _ = unscented_kalman_filter(measurements, process_variance, measurement_variance)
comp_estimates = complementary_filter(measurements)
pf_estimates = particle_filter(measurements)


# Оцінка точності всіх фільтрів з відсотковим представленням
def calculate_metrics(true_position, estimates):
    mae = np.mean(np.abs(true_position - estimates))
    rmse = np.sqrt(np.mean((true_position - estimates) ** 2))
    relative_mae_percent = (mae / np.mean(np.abs(true_position))) * 100
    return mae, rmse, relative_mae_percent


comp_mae, comp_rmse, comp_relative_mae_percent = calculate_metrics(true_position, comp_estimates)
pf_mae, pf_rmse, pf_relative_mae_percent = calculate_metrics(true_position, pf_estimates)

kalman_mae, kalman_rmse, kalman_relative_mae_percent = calculate_metrics(true_position, kalman_estimates)
ekf_mae, ekf_rmse, ekf_relative_mae_percent = calculate_metrics(true_position, ekf_estimates)
ukf_mae, ukf_rmse, ukf_relative_mae_percent = calculate_metrics(true_position, ukf_estimates)

# Вивід результатів у консоль
print("Порівняння продуктивності фільтрів:")
print("Комплементарний фільтр:")
print(f"  Середня абсолютна помилка (MAE): {comp_mae:.3f}")
print(f"  Середньоквадратична помилка (RMSE): {comp_rmse:.3f}")
print(f"  Відносна похибка у відсотковому значенні: {comp_relative_mae_percent:.2f}%")
print("Частинковий фільтр:")
print(f"  Середня абсолютна помилка (MAE): {pf_mae:.3f}")
print(f"  Середньоквадратична помилка (RMSE): {pf_rmse:.3f}")
print(f"  Відносна похибка у відсотковому значенні: {pf_relative_mae_percent:.2f}%")
print("Стандартний фільтр Калмана:")
print(f"  Середня абсолютна помилка (MAE): {kalman_mae:.3f}")
print(f"  Середньоквадратична помилка (RMSE): {kalman_rmse:.3f}")
print(f"  Відносна похибка у відсотковому значенні: {kalman_relative_mae_percent:.2f}%")
print("Розширений фільтр Калмана:")
print(f"  Середня абсолютна помилка (MAE): {ekf_mae:.3f}")
print(f"  Середньоквадратична помилка (RMSE): {ekf_rmse:.3f}")
print(f"  Відносна похибка у відсотковому значенні: {ekf_relative_mae_percent:.2f}%")
print("Беззапаховий фільтр Калмана:")
print(f"  Середня абсолютна помилка (MAE): {ukf_mae:.3f}")
print(f"  Середньоквадратична помилка (RMSE): {ukf_rmse:.3f}")
print(f"  Відносна похибка у відсотковому значенні: {ukf_relative_mae_percent:.2f}%")

# Графік істинного положення та вимірювань
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(measurements, label='Measurements', color='gray', alpha=0.5)
plt.title("True Position and Measurements")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Графік для Комплементарного фільтру
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(comp_estimates, label='Complementary Filter Estimate', color='purple')
plt.title("Complementary Filter")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Графік для Частинкового фільтру
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(pf_estimates, label='Particle Filter Estimate', color='orange')
plt.title("Particle Filter")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Графік для Стандартного фільтру Калмана
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(kalman_estimates, label='Kalman Filter Estimate', color='blue')
plt.title("Kalman Filter")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Графік для Розширеного фільтру Калмана
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(ekf_estimates, label='Extended Kalman Filter Estimate', color='green')
plt.title("Extended Kalman Filter")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Графік для Беззапахового фільтру Калмана
plt.figure()
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(ukf_estimates, label='Unscented Kalman Filter Estimate', color='red')
plt.title("Unscented Kalman Filter")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.legend()
plt.show()

# Візуалізація результатів
plt.figure(figsize=(15, 10))
plt.plot(true_position, label='True Position', linestyle='--', color='black')
plt.plot(measurements, label='Measurements', color='gray', alpha=0.5)
plt.plot(kalman_estimates, label='Kalman Filter', color='blue')
plt.plot(ekf_estimates, label='Extended Kalman Filter', color='green')
plt.plot(ukf_estimates, label='Unscented Kalman Filter', color='red')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.legend()
plt.title('Estimation of Robot Position using Different Kalman Filters')
plt.show()
