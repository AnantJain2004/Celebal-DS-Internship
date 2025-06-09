# Assignment 1 - Create lower triangular, upper triangular and pyramid containing the "*" character.

def lower_triangular(n):
    print("\nLower Triangular Pattern:")
    for i in range(1, n + 1):               # Rows
        for j in range(i):                  # Columns = current row index
            print("*", end="")
        print()


def upper_triangular(n):
    print("\nUpper Triangular Pattern:")
    for i in range(n):                      # Rows
        for j in range(i):                  # Spaces before stars
            print(" ", end="")
        for k in range(n - i):              # Stars
            print("*", end="")
        print()


def pyramid(n):
    print("\nPyramid Pattern:")
    for i in range(n):                      # Rows
        for j in range(n - i - 1):          # Spaces before stars
            print(" ", end="")
        for k in range(2 * i + 1):          # Stars
            print("*", end="")
        print()


# Taking input from the user
try:
    rows = int(input("Enter number of rows: "))
    if rows <= 0:
        print("Please enter a positive number.")
    else:
        lower_triangular(rows)
        upper_triangular(rows)
        pyramid(rows)

except ValueError:
    print("Invalid input. Please enter a number.")