

class TrafficPredictor:
    @staticmethod
    def predict_duration(queue, density, occupancy, speed):
        """
        Decide green light duration based on traffic metrics.
        Returns 10, 20, 30, or 40 seconds.
        """
        score = 0.6 * queue + 2.0 * density + 0.1 * occupancy - 0.3 * (speed / 3.6)

        if score > 15:
            return 40
        elif score > 10:
            return 30
        elif score > 5:
            return 20
        else:
            return 10
