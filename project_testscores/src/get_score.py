def get_score():
    """This function should prompt the user to enter a test score between 0 and 100."""
    score = float(input("Enter a test score (0-100): "))
    if score < 0 or score > 100:
        print("Invalid score. Please enter a score between 0 and 100.")
        return get_score()
    return score


if __name__ == "__main__":
    # Example usage
    score = get_score()
    print(f"You entered a score of: {score}")
