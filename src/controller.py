import numpy as np

class Controller:
    def __init__(self, omega: float = 5*np.pi, phase_coefficent: float = -2, amplitude: float = 0.5, offset: float = 1.0) -> None:
        self.omega = omega
        self.phase_coefficent = phase_coefficent
        self.amplitude = amplitude
        self.offset = offset
    
    def sinus_signal(self, t: float, muscle_indices_x: np.ndarray) -> np.ndarray:
        compute_phases = self.phase_coefficent * muscle_indices_x
        return np.sin(self.omega * t - compute_phases)

    def action_signal(self, t: float, muscle_indices_x: np.ndarray) -> np.ndarray:
        return self.amplitude * self.sinus_signal(t, muscle_indices_x) + self.offset