"""
Controls geometry elements.

Include geometry classes such as:
- Point
- LineSegment
- Circle
as well as
- Canvas - for drawing objects on

Also handles distance calculations, collision detection, and
closest position for collision prevention between circle and line.

Mover class from display.py which controls Predator and Prey
inherits from Circle class.
"""


import math
from typing import Union, List, Optional, Tuple, Sequence

import numpy as np
from abc import ABC, abstractmethod
from skimage import draw
import cv2

# from utils import clamp


# from envs.display import Predator

class Geometry(ABC):

    @abstractmethod
    def draw_on(self, canvas: 'Canvas'):
        ...


class Point(Geometry):

    def __init__(self, x: float=0, y: float=0):
        self.x = x
        self.y = y

    def get_xy(self):
        return self.x, self.y

    def draw_on(self, canvas):
        pass

    def move_point(self, angle: float, distance: float, inplace: bool=False) -> Optional['Point']:
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        if not inplace:
            return Point(self.x + x, self.y + y)
        else:
            self.x += x
            self.y += y
            return None

    def __str__(self):
        return "({}, {})".format(round(self.x, 2), round(self.y, 2))

    def __repr__(self):
        return "Point{}".format(str(self))

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)


class InfLine(Geometry):
    """
    Infinite line defined algebraically by ax + by + c = 0

    Contains methods such as:
    - substitute(x or y)
    - intersect
    - closest point on a line
    - closest object
    """

    name = "obstacle"

    def __init__(self, a, b, c):
        """
        ax + by + c = 0
        """
        self.c = c
        self.b = b
        self.a = a

        try:
            self.slope = - self.a / self.b
        except ZeroDivisionError:
            self.slope = np.nan

        try:
            self.intercept = - self.c / self.b
        except ZeroDivisionError:
            self.intercept = np.nan

    @classmethod
    def from_slope(cls, slope: float, point: Point):
        # x1, y1 = point.get_xy()
        # a = slope
        # b = -1
        # c = y1 - slope * x1
        a, b, c = cls._convert_slope(slope, point)

        return cls(a, b, c)

    @classmethod
    def from_angle(cls, angle: float, point: Point):
        """
        Create an infinite line from an angle in radians, and a point.
        """
        slope = np.tan(angle)
        return cls.from_slope(slope, point)

    def get_coefs(self) -> tuple[float, float, float]:
        return self.a, self.b, self.c

    def substitute(self, x=None, y=None):
        try:
            if x is None and y is None:
                raise Exception("Both x and y cannot be missing.")
            elif y is None:
                if self.a == 0:
                    return - self.c / self.b
                elif self.b == 0:
                    return np.nan
                else:
                    y = - (self.a * x + self.c) / self.b
                return y
            elif x is None:
                if self.a == 0:
                    return np.nan
                elif self.b == 0:
                    return - self.c / self.a
                else:
                    x = - (self.b * y + self.c) / self.a
                return x
        except ZeroDivisionError:
            return np.nan

    def intersect(self, other: 'InfLine') -> Point:
        """
        Return an intersection point between two lines.
        """
        # Line equations ax + by + c = 0
        a1, b1, c1 = self.get_coefs()
        a2, b2, c2 = other.get_coefs()

        # Using homogeneous coordinates (a,b,c)
        c = a1 * b2 - a2 * b1
        a = (b1 * c2 - b2 * c1)
        b = (a2 * c1 - a1 * c2)

        if c == 0:
            return None

        # (x,y) = (a/c, b/c)
        x = a/c
        y = b/c

        # x = np.float32(x)
        # y = np.float32(y)

        return Point(x, y)

    def draw_on(self, canvas):
        ends = []
        for bound in canvas.boundaries:
            p = self.intersect(bound)
            if p:
                ends.append(p)

        segment = LineSegment(ends[0], ends[1])
        return segment.draw_on(canvas)

    @classmethod
    def _convert_slope(self, slope: float, point: Point) -> tuple:
        """
        Convert a slope and a point into (a,b,c) coefficients in the form
        ax + by + c = 0
        """
        x1, y1 = point.get_xy()

        if np.abs(slope) < 1e-8:
            # Slope = 0 ->
            # Horizontal line ( y = -c)
            a = 0
            b = -1
            c = y1
        elif np.abs(slope) > 1e+8:
            # Slope -> Inf
            # Vertical line (x = -c/a)
            a = 1
            b = 0
            c = -x1
        else:
            a = slope
            b = -1
            c = y1 - slope * x1

        return a,b,c

    def __contains__(self, point: Point):
        value_check_x = np.abs(point.y - self.substitute(x=point.x)) <= np.finfo(np.float32).eps
        value_check_y = np.abs(point.x - self.substitute(y=point.y)) <= np.finfo(np.float32).eps
        value_check = value_check_x or value_check_y
        return value_check

    def __str__(self):
        if self.a == 0:
            return f"y = {-self.c / self.b}"
        elif self.b == 0:
            return f"x = {-self.c / self.a}"
        else:
            return f"{self.a}x + {self.b}y + {self.c} = 0"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.a}, {self.b}, {self.c})"


class Ray(InfLine):
    """
    A line which has one starting point but extends into infinity.
    """
    def __init__(self, angle: float, origin: Point):
        slope = np.tan(angle)
        a, b, c = super()._convert_slope(slope, origin)

        super().__init__(a, b, c)
        self.origin = origin
        self.angle = angle

    def raycast(self, object_list: List[Geometry],
                return_ray: Optional[bool] = False) -> \
            Tuple[Optional[float], Optional[Union['LineSegment', 'Ray']], Optional[str]]:
        """
        Perform a raycasting of a ray to a list of obstacles (lines) and return the distance
        to the nearest obstacle that intersects with the ray.
        """
        dist, ray, obj_type = None, None, None
        distances = []
        # Store points if returning rays
        points = []
        collided_objects = []
        for obj in object_list:
            P = self.intersect(obj)
            if P:  # If intersect
                dist = distance_point_to_point(self.origin, P)
                distances.append(dist)
                collided_objects.append(obj)
                if return_ray:
                    points.append(P)
        collided = len(distances) > 0

        if collided:
            dist = min(distances)
            idx = np.argmin(distances)
            obj_type = collided_objects[idx].name
        # else:
        #     dist = None

        if return_ray and collided:
            idx = np.argmin(distances)
            P = points[idx]
            ray = LineSegment(self.origin, P)
            # return (dist, line)
        elif return_ray and not collided:
            ray = self
            # return (dist, self)
        # else:
        #     return (dist, None)
        #
        return (dist, ray, obj_type)


    def intersect(self, other: Union[Geometry]) -> Union[Point, None]:
        if isinstance(other, InfLine):
            P = super().intersect(other)

            if P is not None and P in self and P in other:
                return P
            else:
                return None
        elif isinstance(other, Circle):
            P = other.intersect(self)

            if P is not None:
                if isinstance(P, tuple):
                    dists = [distance_point_to_point(self.origin, point) for point in P]
                    P = P[np.argmin(dists)]

                if P in self:
                    return P
                else:
                    return None
            else:
                return None

    def draw_on(self, canvas: 'Canvas'):
        end = Point()
        for bound in canvas.boundaries:
            p = self.intersect(bound)
            if p:
                end = p
                break
        segment = LineSegment(self.origin, end)
        return segment.draw_on(canvas)

    def _is_in_same_quadrant(self, point: Point):
        expected_quadrant = np.array(quadrant(self.angle))
        actual_quadrant = np.sign(safe_round((point - self.origin).get_xy()))
            # Needs rounding to prevent floating imprecision
        return np.all(expected_quadrant == actual_quadrant)

    def __contains__(self, point: Point):
        range_check = self._is_in_same_quadrant(point)
        value_check = super().__contains__(point)
        check = range_check & value_check
        return check

    def __repr__(self):
        return f"{self.__class__.__name__}({self.angle}, {self.origin})"

    def __str__(self):
        return f"{super().__str__()} at {self.origin}, {np.rad2deg(self.angle):.0f} degrees"


class LineSegment(InfLine):

    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2
        self.end_points = (self.point1, self.point2)
        self.length = distance_point_to_point(point1, point2)
        slope = slope_between_points(point1, point2)

        a, b, c = super()._convert_slope(slope, point1)
        super().__init__(a,b,c)
        # self.intercept = self.point1.y - self.slope * self.point1.x
        self.angle = np.arctan(self.slope)

    # def substitute(self, x=None, y=None):
    #     if x is None and y is None:
    #         raise Exception("Both x and y cannot be missing.")
    #     elif y is None:
    #         y = self.slope * x + self.intercept
    #         return y
    #     elif x is None:
    #         x = (y - self.intercept) / self.slope
    #         return x

    # def __contains__(self, item: Point):
    #     if self.point1.x <= item.x <= self.point2.x and \
    #         self.point1.y

    def __contains__(self, point: Point):
        """
        Checks if a point lies in this line segment.
        """
        # First check if the point algebraically lies on the line
        # lies_on_line = np.abs(point.y - self.substitute(x=point.x)) <= np.finfo(np.float32).eps
        value_check = super().__contains__(point)
        # Then check if the point is within the same (x,y) range as the line segment
        inside_x = min(self.point1.x, self.point2.x) <= point.x <= max(self.point1.x, self.point2.x)
        inside_y = min(self.point1.y, self.point2.y) <= point.y <= max(self.point1.y, self.point2.y)
        range_check = inside_x and inside_y
        check = value_check & range_check
        return check

    def draw_on(self, canvas: 'Canvas'):
        xx, yy = draw.line(
            int(self.point1.x), int(self.point1.y),
            int(self.point2.x), int(self.point2.y)
        )
        np.clip(xx, canvas.xmin, canvas.width - 1, out=xx)
        np.clip(yy, canvas.ymin, canvas.height - 1, out=yy)
        canvas.canvas[xx, yy] = 0
        return canvas.canvas

    # def __str__(self):
    #     return "y = {m}x + {b}".format(m=self.slope, b=self.intercept)

    def __repr__(self):
        if self.point1.x <= self.point2.x:
            return f"LineSegment({self.point1}, {self.point2})"
        else:
            return f"LineSegment({self.point2}, {self.point1})"


class Rectangle(LineSegment):
    """
    A collection of four line segments that form an enclosing rectangle.
    """

    def __init__(self, angle: float, center: Point, width: float, height: float):
        pass

    @classmethod
    def from_segment(cls, line: LineSegment, thickness: float):
        return cls



class Circle(Geometry):
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    # @classmethod
    # def from_coords(cls, x, y, radius):
    #     """Accepts x,y coordinates as separate parameters instead of
    #     point objects."""
    #     return cls(Point(x,y), radius)

    def draw_on(self, canvas: 'Canvas'):
        xx,yy,val = draw.circle_perimeter_aa(
            int(self.x), int(self.y), int(self.radius),
        shape=canvas.canvas.shape)

        canvas.canvas[xx,yy] = 1 - val
        return canvas.canvas

    def radial_raycast(self,
                       obj_list: List[Geometry],
                       canvas: 'Canvas', num_rays: Optional[int] = 8,
                       return_rays: Optional[bool] = False) -> \
            Tuple[
                np.ndarray[float],
                Optional[List[LineSegment]],
                Optional[List[str]]
            ]:
        """
        Perform a series of raycasting measurements in a counterclockwise loop around the circle center.
        Requires a list of obstacles (lines) and a canvas (for the boundaries), as well as an optional
        control over the number of radial splits.
        Returns a list of distances
        """
        center = Point(self.x, self.y)
        rays = create_radial_rays(origin=center, num_splits=num_rays)

        crossed_dist = []
        crossed_rays = []
        crossed_obj = []
        for ray in rays:
            # crossed_lines = []
            distance, line, obj_type = ray.raycast(obj_list, return_ray=return_rays)

            if distance is None:
                distance, line, obj_type = ray.raycast(canvas.boundaries, return_ray=return_rays)

            if distance is None:
                distance = np.nan

            crossed_dist.append(distance)
            if return_rays:
                crossed_rays.append(line)
            crossed_obj.append(obj_type)

        crossed_dist = np.array(crossed_dist)

        return crossed_dist, crossed_rays, crossed_obj

    def intersect(self, other: 'InfLine') -> Union[None, Point, Tuple[Point]]:
        """
        Detect intersection of a circle with a line.
        Algorithm from
        https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle
        """
        # Vertical line
        if other.b == 0:
            # Solve quadratic equations where both circle and linie are true
            a = other.a ** 2 + other.b ** 2
            b = 2 * other.b * other.c + 2 * other.a * other.b * self.x - 2 * other.a ** 2 * self.y
            c = other.c ** 2 + 2 * other.a * other.c * self.x - \
                other.a ** 2 * (self.radius ** 2 - self.x ** 2 - self.y ** 2)

            discriminant = b ** 2 - 4 * a * c

            if discriminant < 0:
                return None
            elif discriminant == 0:
                y = - b / (2 * a)
                x = other.substitute(y=y)
                return Point(x, y)
            else:
                y = [- (b + sign * np.sqrt(discriminant)) / (2 * a) for sign in (-1, 1)]
                x = [other.substitute(y=val) for val in y]
                return tuple(Point(xx, yy) for xx, yy in zip(x, y))
        else:
            # Solve quadratic equations where both circle and linie are true
            a = other.a**2 + other.b**2
            b = 2 * other.a * other.c + 2 * other.a * other.b * self.y - 2 * other.b **2 * self.x
            c = other.c **2 + 2 * other.b * other.c * self.y - \
                other.b **2 * (self.radius**2 - self.x **2 - self.y **2)

            discriminant = b**2 - 4 * a * c

            if discriminant < 0:
                return None
            elif discriminant == 0:
                x = - b / (2 * a)
                y = other.substitute(x)
                return Point(x,y)
            else:
                x = [- (b + sign * np.sqrt(discriminant)) / (2 * a) for sign in (-1, 1)]
                y = [other.substitute(x=val) for val in x]
                return tuple(Point(xx,yy) for xx, yy in zip(x, y))

        return None

    def detect_collision(self, line: LineSegment):
        center = Point(self.x, self.y)
        # First check if passed line segment ends
        # if any([distance_point_to_point(center, end_point) > line.length for end_point in (line.point1, line.point2)]):
        if self.is_clear_of_line(line):
            return False
        # Otherwise check if radius is less than distance
        else:
            d = distance_line_point(line, center)
            return d <= self.radius

    def direction_from(self, line: LineSegment):
        """
        Return (x, y) unit vector telling the direction of this object is in relation
        to another object (e.g. a line).

        For example, if this circle is below and right of a line, this method will return
        (+1.0, -1.0).
        """
        sign_y = np.sign(self.y - line.substitute(x=self.x))
        sign_x = np.sign(self.x - line.substitute(y=self.y))

        return sign_x, sign_y

    def closest_position_to_line(self, line: LineSegment) -> Union[Point, None]:
        """
        Returns the point closest to where the circle can safely touch the line without overlapping.

        """
        if self.is_clear_of_line(line):
            return None

        # Determines whether circle is above or underneath the line
        sign_x, sign_y = self.direction_from(line)

        # Closest point ON the line
        P = closest_point_on_line(Point(self.x, self.y), line)

        # Deviation from P to safe point G
        x = sign_x * self.radius * np.sin(np.abs(line.angle))
        y = sign_y * self.radius * np.cos(np.abs(line.angle))

        # Add deviation to point P to get G
        G = P + Point(x,y)

        return G

    def is_clear_of_line(self, line: LineSegment):
        """Check whether circle is in a position where the closest perpendicular point lies
        outside line segment."""

        center = Point(self.x, self.y)
        P = closest_point_on_line(center, line)
        return P not in line

        # # Previous implementation
        # center = Point(self.x, self.y)
        # dist = max(*(distance_point_to_point(center, end) for end in line.end_points))
        # is_clear = (dist) >= (line.length + self.radius)
        # return is_clear

    def distance_to_line(self, line: LineSegment) -> float:
        """
        Closest distance to specified line, taking into account lines that are clear of sight,
        by returning Inf instead.
        """
        if self.is_clear_of_line(line):
            return np.Inf
        else:
            return distance_point_to_line(Point(self.x, self.y), line)


class Canvas:
    """
    Canvas for drawing objects using openCV
    """

    boundaries: list[LineSegment]

    def __init__(self, width: int, height: int = None,
                 name: str = "Environment"):
        if height is None:
            height = width

        self.width = width
        self.height = height
        self.xmin = 0
        self.ymin = 0

        self.canvas = np.ones((self.height, self.width))
        self.shape = self.canvas.shape
        self.name = name

        self._create_boundaries()

    def _create_boundaries(self):
        self.boundaries = []
        corners = [Point(0,0), Point(0, self.height), Point(self.width, self.height), Point(self.width, 0)]
        lagged  = corners[1:] + [corners[0]]
        for p1, p2 in zip(corners, lagged):
            bound = LineSegment(p1, p2)
            self.boundaries.append(bound)

    def draw(self, obj: Union[Geometry, List[Geometry]]):
        if isinstance(obj, List):
            for item in obj:
                self.canvas = item.draw_on(self)
        else:
            self.canvas = obj.draw_on(self)

    def show(self, frame_delay: int=0, manual=False):
        # Transform [i,j] indexing to (x,y) coordinates
        cartesian = np.transpose(np.flip(self.canvas, 1))
        cv2.imshow(self.name, cartesian)
        if manual:
            key = cv2.waitKeyEx(0)
        else:
            key = cv2.waitKey(frame_delay)
        return key

    def _force_clear(self):
        self.canvas = np.ones((self.height, self.width))

    def clear(self):
        self.canvas = np.ones((self.height, self.width))

    def close(self):
        cv2.destroyAllWindows()

# Add collisions

def distance_line_point(line: LineSegment, point: Point):
    x0, y0 = point.get_xy()
    x1, y1 = line.point1.get_xy()
    x2, y2 = line.point2.get_xy()

    numer = abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1))
    denom = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    d = numer / denom
    return d


def distance_point_to_line(point: Point, line: LineSegment):
    return distance_line_point(line, point)


def distance_point_to_point(point1: Point, point2: Point):
    """Euclidean distance between two points"""
    x1, y1 = point1.get_xy()
    x2, y2 = point2.get_xy()

    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d


def slope_between_points(point1: Point, point2: Point):
    delta_x = point2.x - point1.x
    delta_y = point2.y - point1.y
    try:
        return delta_y / delta_x
    except ZeroDivisionError:
        return np.Inf


def closest_point_on_line(point: Point, line: LineSegment):
    # Find perpendicular line
    perpendicular_slope = -1/(line.slope)
    perpendicular_intercept = point.y - perpendicular_slope * point.x # y=mx+b -> b=y-mx

    # Find intersecting point
    x = (perpendicular_intercept - line.intercept) / (line.slope - perpendicular_slope)
        # a1x + b1 = a2x + b2
        # x = (b2-b1) / (a1-a2)
    y = line.substitute(x=x)

    return Point(x, y)


def generate_random_line(canvas_shape: tuple[int, int], seed=None) -> LineSegment:
    width = canvas_shape[1]
    height = canvas_shape[0]
    np.random.seed(seed)
    p1 = Point(np.random.randint(width), np.random.randint(height))
    p2 = Point(np.random.randint(width), np.random.randint(height))
    line = LineSegment(p1, p2)
    return line


def quadrant(angle) -> tuple:
    """
    Return the (x,y) sign of the angle (radians), showing which quadrant the angle is in.
    """
    # x = np.sign(np.round(np.cos(angle), 8))
    # y = np.sign(np.round(np.sin(angle), 8))
    x = np.sign(safe_round(np.cos(angle)))
    y = np.sign(safe_round(np.sin(angle)))
    return x, y


def safe_round(value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if isinstance(value, float):
        precision = np.finfo(value).precision
    elif isinstance(value, tuple):
        precision = np.finfo(np.array(value).dtype).precision
    elif isinstance(value, np.ndarray):
        precision = np.finfo(value.dtype).precision
    else:
        precision = np.finfo(np.float32).precision
    return np.round(value, precision)

def create_radial_rays(origin: Point, num_splits: int) -> List[Ray]:
    """
    Create a list of Rays radiating from a single origin at constant angular intervals.
    """
    assert isinstance(num_splits, int) and num_splits > 0, "Number of radial splits cannot be non-positive integers"
    angle = 2 * np.pi / num_splits
    rays = [Ray(i * angle, origin) for i in range(num_splits)]
    return rays


def create_perpendicular_line(point: Point,
                              angle: Optional[float],
                              line: Optional[InfLine],
                              length: Optional[float]) -> Union[Ray, LineSegment]:
    pass


# def create_box(point1: Point, point2: Point, point3: Point, point4: Point) -> List[LineSegment]:
def create_polygon(*args: Point) -> List[LineSegment]:
    corners = [*args]
    lagged = corners[1:] + [corners[0]]
    borders = [LineSegment(p1, p2) for p1,p2 in zip(corners, lagged)]
    return borders


def convert_line_to_box(line: LineSegment, width: float):
    w = width
    l = line.length
    angle = line.angle
    if line.point1.x <= line.point2.x:
        p1 = line.point1
        p2 = line.point2
    else:
        p1 = line.point2
        p2 = line.point1
    angle -= np.pi / 2
    p3 = p2.move_point(angle, w)
    angle -= np.pi / 2
    p4 = p3.move_point(angle, l)
    box = create_polygon(p1, p2, p3, p4)
    return box


def read_obstacle_config(file: str, canvas: Canvas) -> List[LineSegment]:
    data = np.genfromtxt(file, delimiter=",")
    obstacles = []
    for x1, y1, x2, y2, width in data:
        line = LineSegment(Point(x1 * canvas.width, y1 * canvas.height),
                           Point(x2 * canvas.width, y2 * canvas.height))
        lines = convert_line_to_box(line, width * canvas.width)
        obstacles.extend(lines)
    return obstacles


if __name__ == "__main__":
    point = Point(1, 1)

    W = 500
    canvas = Canvas(W, W)

    line = LineSegment(Point(0.3 * W, 0.8 * W), Point(0.7 * W, 0.1 * W))
    # circle = Circle(Point(0.7*W, 0.9*W), 0.1*W)
    circle = Circle.from_coords(0.5*W, 0.5*W, 0.1*W)

    print(circle.detect_collision(line))

    # circle = BallCollider((W,W))

    canvas.draw(line)
    canvas.draw(circle)

    canvas.show(0)

    canvas.close()
