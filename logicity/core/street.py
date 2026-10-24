from .config import *

class Street:
    def __init__(self, position=[0, 0], length=1, orientation='horizontal', type='Traffic Street', directions=2, width=0):
        """
        Initialize a new street.

        :param position: A tuple (x, y) representing the start point of the street in the grid.
        :param length: Length of the street on the grid.
        :param orientation: 'horizontal' or 'vertical' indicating street's direction.
        :param street_type: 'Walking Street' or 'Traffic Street' to distinguish between walking streets and traffic streets.
        :param directions: 1 for one-way streets and 2 for two-way streets.
        """
        self.position = position
        self.length = length
        self.orientation = orientation
        self.type = type
        self.directions = directions

        # Depending on the street type, we can assign a default width.
        if width == 0:
            if type == 'Walking Street':
                self.width = WALKING_STREET_WID  # or any default width for walking streets
            else:
                self.width = TRAFFIC_STREET_WID  # or any default width for traffic streets
        else:
            self.width = width

    def contains_point(self, x, y):
        """Check if a point (x, y) is within the street's boundaries."""
        if self.orientation == 'horizontal':
            return (self.position[0] <= x < self.position[0] + self.width) and \
                   (self.position[1] <= y < self.position[1] + self.length)
        else:
            return (self.position[0] <= x < self.position[0] + self.length) and \
                   (self.position[1] <= y < self.position[1] + self.width)

    def is_one_way(self):
        """Check if the street is one-way."""
        return self.directions == 1

    def __str__(self):
        """String representation for debugging purposes."""
        return f"Street(Position: {self.position}, Length: {self.length}, Type: {self.street_type}, Directions: {self.directions})"
