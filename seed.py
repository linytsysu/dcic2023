import pandas as pd

seeds = [0, 1, 2, 42, 2023, 3407, 10, 100, 1000, 10000, 3, 5, 7, 11, 13, 17, 23, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
seeds = [3, 5, 7, 11, 13, 17, 23, 31, 37,]

res = []
for seed in seeds:
    pred = pd.read_pickle(f'test_res_{seed}.pkl')
    res.append(pred.set_index('zhdh'))

res = pd.concat(res, axis=1)
res = res.mean(axis=1).reset_index()
res[0] = res[0].apply(lambda x: (x > 0.5) + 0)
res.columns = ['zhdh', 'black_flag']

res2 = pd.read_csv('result.csv')

res2 = res2.merge(res, how='left', on='zhdh')[['zhdh', 'black_flag_y']]
res2.columns = ['zhdh', 'black_flag']
res2.to_csv('final_version.csv', index=False)

# res.to_csv('0321.csv', index=False)
