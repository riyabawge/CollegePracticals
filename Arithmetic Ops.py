import statistics

numbers = [10, 20, 30, 40, 50]

print("Numbers:", numbers)

total = sum(numbers)
average = total / len(numbers)
minimum = min(numbers)
maximum = max(numbers)

print("\nArithmetic Calculations")
print("Sum:", total)
print("Average:", average)
print("Minimum:", minimum)
print("Maximum:", maximum)

mean = statistics.mean(numbers)
median = statistics.median(numbers)
std = statistics.stdev(numbers)

print("\nStatistical Calculations")
print("Mean:", mean)
print("Median:", median)
print("Standard Deviation:", round(std, 2))