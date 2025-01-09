import matplotlib.pyplot as plt
from math import e, atan2
import numpy as np
from matplotlib.animation import FuncAnimation

SCREEN_X_SIZE = 500
SCREEN_Y_SIZE = 500
SCALE = 5
FRAME_STEP = 20


def mod(z: complex) -> float:
    return (z.real * z.real + z.imag * z.imag) ** 0.5


def arg(z: complex) -> float:
    return atan2(z.imag, z.real)


def zeros_transition(z: np.ndarray) -> np.ndarray:
    return z


def first_transition(z: np.ndarray) -> np.ndarray:
    v = 1j * (np.log(np.abs(z / e)) + 1j * np.angle(z / e))
    return v


def second_transition(z: np.ndarray) -> np.ndarray:
    v = first_transition(z)
    arg_v = np.angle(v)
    mod_v = np.abs(v)
    g = mod_v * np.exp(1j * arg_v / 2)
    return g


def third_transition(z: np.ndarray) -> np.ndarray:
    g = second_transition(z)
    h = 1j * g / np.pi + 1
    return h


def fourth_transition(z: np.ndarray) -> np.ndarray:
    h = third_transition(z)
    h1 = -2 / h / 10
    return h1


def fifth_transition(z: np.ndarray) -> np.ndarray:
    h1 = fourth_transition(z)
    arg_h = np.angle(h1)
    mod_h = np.abs(h1)
    h2 = mod_h * np.exp(1j * (arg_h - np.pi / 2))
    return h2


def calculate_to_frame(z: np.ndarray, frame: int, transitions_) -> np.ndarray:
    current_transition = frame // FRAME_STEP
    current_number = transitions_[current_transition](z)
    if current_transition < len(transitions_) - 1:
        t = frame % FRAME_STEP / FRAME_STEP
        current_number = (1 - t) * current_number + t * transitions_[current_transition + 1](z)
    return current_number


def initial_point_set(z: complex) -> bool:
    return mod(z) > e


def convert_from_image_coordinates_to_complex_number(i: int, j: int) -> complex:
    x = (j / float(SCREEN_X_SIZE) - 0.5) * float(SCALE) * 2.0
    y = (i / float(SCREEN_Y_SIZE) - 0.5) * float(SCALE) * -2.0
    return x + y * 1j


def filler(point_set_: list[complex]) -> None:
    for i in range(0, SCREEN_Y_SIZE, 2):
        for j in range(0, SCREEN_X_SIZE, 2):
            complex_point = convert_from_image_coordinates_to_complex_number(i, j)
            if initial_point_set(complex_point):
                point_set_.append(complex_point)


def init_render():
    first_frame_point_set = []
    filler(first_frame_point_set)

    fig, ax = plt.subplots()
    sc = ax.scatter([p.real for p in first_frame_point_set], [p.imag for p in first_frame_point_set], s=1)

    point_set = [i for i in first_frame_point_set]

    first_frame_point_set = np.array(first_frame_point_set, dtype=np.complex128)

    transitions = [zeros_transition, first_transition, second_transition, third_transition, fourth_transition,
                   fifth_transition]
    transitions += transitions[::-1]

    def update(frame):
        global point_set
        point_set = calculate_to_frame(first_frame_point_set, frame, transitions)
        sc.set_offsets(np.c_[[p.real for p in point_set], [p.imag for p in point_set]])
        return sc,

    ani = FuncAnimation(fig, update, frames=len(transitions) * FRAME_STEP, interval=5, blit=True)

    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    init_render()
