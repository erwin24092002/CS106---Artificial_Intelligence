# Import the libraries
from ortools.algorithms import pywrapknapsack_solver 
import time 
import pandas as pd

testCases = ['n00050_R01000_s000', 'n00100_R01000_s000' ,'n00200_R01000_s000', 
             'n00500_R01000_s000', 'n01000_R01000_s000']
groups = ['00Uncorrelated','01WeaklyCorrelated','02StronglyCorrelated',
              '03InverseStronglyCorrelated', '04AlmostStronglyCorrelated', '05SubsetSum', 
              '06UncorrelatedWithSimilarWeights', '07SpannerUncorrelated', 
              '08SpannerWeaklyCorrelated','09SpannerStronglyCorrelated', 
              '10MultipleStronglyCorrelated','11ProfitCeiling', '12Circle']

# Declare the solver
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

# Solve
data = []  # to save csv_result
for group in groups: 
    for testCase in testCases: 
        # Create data
        inputPath = 'testGroups/'+group+'/'+testCase+'.kp' 
        values = []
        weights = [[]]
        capacities = []
        with open(inputPath) as f:
            lines = f.read().splitlines()
            n = int(lines[1])
            capacities.append(int(lines[2]))
            for i in range(4, 4+n):
                item = lines[i].split()  #(item = [value, weight])
                values.append(int(item[0]))
                weights[0].append(int(item[1]))
            f.close()
                
        # Call the solver 
        solver.Init(values, weights, capacities)
        time_limit = 180.0
        solver.set_time_limit(time_limit)
        
        time_start = time.time()
        computed_value = solver.Solve()   # Total value
        time_end = time.time()
        
        packed_items = []
        packed_weights = []
        total_weight = 0
        for i in range(len(values)):
            if solver.BestSolutionContains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
                
        # Save txt_result
        outputPath = 'result/txt/' + group + '_' + testCase +'.txt' 
        with open(outputPath, 'w') as f: 
            f.write('Total value: ' + str(computed_value) + '\n')
            f.write('Total weight: ' + str(total_weight) + '\n') 
            f.write('Runtime: ' + str(time_end-time_start) + '\n') 
            f.write('Optimal: ' + str(time_end-time_start<=time_limit) + '\n') 
            f.write('Strategy: \n')
            f.write('>>Packed items:'+ str(packed_items) + '\n')
            f.write('>>Packed weights:'+ str(packed_weights) + '\n')
            f.close()
            
        # Upgrade csv_result
        data.append([group, testCase, computed_value, total_weight, (time_end-time_start), round((time_end-time_start<=time_limit),5)])
    
    # Done 
    print('Completed ' + group)

# Save csv_result
columns = ['Group test', 'Case', 'Total value', 'Total weight', 'Runtime', 'Optimal'] 
df = pd.DataFrame(data, columns=columns)
df.to_csv('result/csv/result.csv')


