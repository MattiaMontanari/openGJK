from pyopengjk import compute_minimum_distance, Point3


def main():
    vertices0 = [
        Point3(0.0, 5.5, 0.0),
        Point3(2.3, 1.0, -2.0),
        Point3(8.1, 4.0, 2.4),
        Point3(4.3, 5.0, 2.2),
        Point3(2.5, 1.0, 2.3),
        Point3(7.1, 1.0, 2.4),
        Point3(1.0, 1.5, 0.3),
        Point3(3.3, 0.5, 0.3),
        Point3(6.0, 1.4, 0.2)
    ]

    vertices1 = [
        Point3(0.0, -5.5, 0.0),
        Point3(-2.3, -1.0, 2.0),
        Point3(-8.1, -4.0, -2.4),
        Point3(-4.3, -5.0, -2.2),
        Point3(-2.5, -1.0, -2.3),
        Point3(-7.1, -1.0, -2.4),
        Point3(-1.0, -1.5, -0.3),
        Point3(-3.3, -0.5, -0.3),
        Point3(-6.0, -1.4, -0.2)
    ]

    distance, simplex = compute_minimum_distance(vertices0, vertices1)
    print(f"Minimum distance: {distance}")
    print("Witness points:")
    print(simplex.witnesses[0])
    print(simplex.witnesses[1])


if __name__ == "__main__":
    main()
