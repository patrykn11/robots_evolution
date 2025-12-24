import numpy as np

class Controller:
    def __init__(self, omega=5*np.pi, phase_coefficent=-2, amplitude=0.5, offset=1.0):
        self.omega = omega
        self.phase_coefficent = phase_coefficent
        self.amplitude = amplitude
        self.offset = offset
    
    def sinus_signal(self, t, muscle_indices_x):
        compute_phases = self.phase_coefficent * muscle_indices_x
        return np.sin(self.omega * t - compute_phases)

    def action_signal(self, t, muscle_indices_x):
        return self.amplitude * self.sinus_signal(t, muscle_indices_x) + self.offset