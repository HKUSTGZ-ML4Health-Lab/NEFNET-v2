import numpy as np


def process_numbers_console():
    """Get two angles (in degrees) via console input, convert to radians and return as np.float32 array"""

    def get_angle(prompt):
        """Safely get angle input with validation"""
        while True:
            try:
                value = input(f"{prompt} (in degrees): ")
                if value.lower() == 'exit':
                    return None
                angle = float(value)
                return np.float32(angle / 180 * np.pi)  # Convert degrees to radians
            except ValueError:
                print("ERROR: Please enter a valid number! Type 'exit' to quit")

    print("=== Enter two angles (type 'exit' to quit) ===")
    theta = get_angle("Enter θ angle")
    if theta is None: return None

    phi = get_angle("Enter φ angle")
    if phi is None: return None

    query_theta = np.array([[theta, phi]], dtype=np.float32)
    print(f"Result (in radians):\n{query_theta}")
    return query_theta


# Usage example
if __name__ == "__main__":
    result = process_numbers_console()
    if result is not None:
        print("Processed array:", result)