from envs.geometry import Ray, Circle, Point, read_obstacle_config, Canvas, create_radial_rays

ray13 = Ray(1.2762720155208536, Point(511, 249))
ray19 = Ray(1.8653206380689396, Point(511, 249))

circle = Circle(511, 249, radius=28.284271247461902)
canvas = Canvas(800, 800)
canvas.clear()
obstacle_file="test_codes/obstacle-null.csv"
obstacle_list = read_obstacle_config(obstacle_file, canvas)
canvas.draw(obstacle_list)
obstacle_list.extend(canvas.boundaries)

rays = create_radial_rays(Point(circle.x, circle.y), 64)

canvas.draw([ray13, ray19])

for i, ray in enumerate(rays):
    obs, _, obj_types = ray.raycast(obstacle_list)
    if obs is None:
        print(i, obs, obj_types)

canvas.show()