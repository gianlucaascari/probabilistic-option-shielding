
class Building:
    TYPES = ['House', 'Gas Station', 'Office', 'Garage', 'Store', 'School', 'Big School']

    def __init__(self, block=1, position=(0, 0), size=(1, 1), height=1, type='House', render=True):
        self.position = position
        self.block = block
        self.size = size
        self.height = height
        self.render = render
        if type in Building.TYPES:
            self.type = type
        else:
            raise ValueError(f"Invalid building type: {type}. Must be one of {Building.TYPES}")

        # Check if height is given as scalar or matrix
        if isinstance(height, (int, float)):
            # If scalar, create a matrix with the given scalar value
            self.height = [[height for _ in range(self.size[1])] for _ in range(self.size[0])]
        elif isinstance(height, list) and all(isinstance(row, list) for row in height):
            # If matrix, just assign it
            self.height = height
        else:
            raise ValueError("Height should be either a scalar or a matrix (2D list)")

    def get_height_at(self, x, y):
        """Return the height at a specific grid cell within the building."""
        if not self.contains_point(x, y):
            return None  # or raise an exception if you prefer
        return self.height[x - self.position[0]][y - self.position[1]]

    def contains_point(self, x, y):
        """Check if a point (x, y) is within the building's boundaries."""
        return (self.position[0] <= x < self.position[0] + self.size[0]) and \
               (self.position[1] <= y < self.position[1] + self.size[1])

    def get_area(self):
        """Return the building's area on the grid."""
        return self.size[0] * self.size[1]

    def __str__(self):
        """String representation for debugging purposes."""
        return f"Building(Position: {self.position}, Size: {self.size}, Height: varies)"

