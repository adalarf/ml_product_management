def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def describe(name: str, values: list[float]) -> str:
    return f"{name}: n={len(values)}, mean={mean(values):.2f}"


if __name__ == "__main__":
    print(describe("sample", [1.0, 2.5, 4.0]))
