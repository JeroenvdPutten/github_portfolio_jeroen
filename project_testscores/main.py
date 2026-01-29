from src.calc_average import calc_average
from src.get_all_scores import get_all_scores


def main():
    num_scores = int(input("Enter the number of test scores: "))
    all_scores = get_all_scores(num_scores)
    average = calc_average(all_scores)
    print(f"The average test score is {average:.2f}")


if __name__ == "__main__":
    main()
