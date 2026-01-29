from src.find_lowest import find_lowest
from src.get_all_scores import get_all_scores


def calc_average(all_scores):
    """Calculates the average of the n highest scores by dropping the lowest score."""
    lowest = find_lowest(all_scores)
    total = sum(all_scores) - lowest
    return total / (len(all_scores) - 1)


if __name__ == "__main__":
    # Example usage
    scores = get_all_scores(5)
    average = calc_average(scores)
    print(f"The average of the highest scores is: {average:.2f}")
