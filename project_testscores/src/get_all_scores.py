from src.get_score import get_score


def get_all_scores(num_scores):
    """Collects a specified number of test scores from the user."""
    scores = []
    for _ in range(num_scores):
        scores.append(get_score())
    return scores


if __name__ == "__main__":
    # Example usage
    number_of_scores = int(
        input("How many test scores do you want to enter? "))
    all_scores = get_all_scores(number_of_scores)
    print(f"You entered the following scores: {all_scores}")
