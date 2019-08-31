import numpy as np
import pandas as pd

data_csv = pd.read_csv("https://canvas.instructure.com/courses/1668248/files/79840890/download?download_frd=1", names=["escolaridade","salario"], header=None)

def compute_error_for_line_given_points(b, a, points):
    total_err = 0
    for i in range(0, len(points)):
        x = points['escolaridade'][i]
        y = points['salario'][i]
        total_err += (y - (a * x + b)) ** 2
    return total_err

def step_gradient(b_current, a_current, points, learningRate):
    b_gradient = 0
    a_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points['escolaridade'][i]
        y = points['salario'][i]
        b_gradient += -(2/N) * (y - ((a_current * x) + b_current))
        a_gradient += -(2/N) * x * (y - ((a_current * x) + b_current))
        
    new_b = b_current - (learningRate * b_gradient)
    new_a = a_current - (learningRate * a_gradient)
    return [new_b, new_a]

def gradient_descent_runner(points, starting_b, starting_a, learning_rate, num_iterations):
    b = starting_b
    a = starting_a
    rss_values = []
    
    for i in range(num_iterations):
        b, a = step_gradient(b, a, points, learning_rate)
        rss = compute_error_for_line_given_points(b,a, points)
        rss_values.append(rss)
    return [b, a, rss_values]

def run():
  learning_rate = 0.001
  initial_b = 0 
  initial_a = 0
  num_iterations = 1000

  [b, a, rss] = gradient_descent_runner(data_csv, initial_b, initial_a, learning_rate, num_iterations)
  print( a, b)

if __name__ == '__main__':
    run()