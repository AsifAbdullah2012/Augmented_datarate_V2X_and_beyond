import matplotlib.pyplot as plt

from data import S3Measurements


bm = S3Measurements()
list_of_data_rate = bm.get_datarate_measurements()

print('Ideal data rate: ', list_of_data_rate[0],' and predicted data rate: ', list_of_data_rate[1])
print('You can draw here ....')
