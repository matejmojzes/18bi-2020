import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dates = []
data = []
prep_dates = []
maxim = []
minim = []

# load data with pandas
column_names = ["YEAR", "MILK_PRODUCTION"]
df = pd.read_csv("dataBI.csv", names=column_names)

dates = df.YEAR.to_list()
data = df.MILK_PRODUCTION.to_list()

# prepare data for better
for d in dates:
    prep_dates.append(d[2:])

# figure data for each year(15 graphs)

for i in range(0,14):
    max_value = data.index(max(data[12*i:(12*i)+12]))
    # for min value was problem with multiple same data values
    # so i had to make sure to get the min value from right interval
    # should be done for max_value as well, but from look at data, max
    # value is always higher then previous one
    value = min(data[12 * i:(12 * i) + 12])
    vvalue = [a for a,val in enumerate(data) if val == value]
    for v in vvalue:
        if 12*i <= v <= (12 * i)+12:
            min_value = v
            break
    maxim.append(max(data[12*i:(12*i)+12]))
    minim.append(min(data[12 * i:(12 * i) + 12]))
    plt.figure(i)
    plt.scatter(prep_dates[12*i:(12*i)+12], data[12*i:(12*i)+12], label="Výroba mléka")
    plt.scatter(prep_dates[max_value], data[max_value], color='red', label="Nejvyšší výroba")
    plt.scatter(prep_dates[min_value], data[min_value], color='yellow', label="Nejnižší výroba")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)
plt.show()


# plot minimum and maximum over the years

i = range(1,15)
plt.title('Nejvyšší naměřené hodnoty (1962-1975)')
plt.scatter(i,maxim)
plt.show()
plt.title('Nejnižší naměřené hodnoty (1962-1975)')
plt.scatter(i, minim)
plt.show()

# plot everything so we can see the repeat tendencies
plt.title('Graf všech naměřených hodnot (1962-1975)')
plt.plot(dates, data)
plt.show()

# conclusion

print("Z grafů jednotlivých roků měření můžeme vidět, že data mají opakující tendenci s každoročním navýšením.")
print("Každý rok výroba narůstá - tím pádem i spotřeba. Jednou z příčin tohoto vývoje je pravděpodobně nárůst populace.")



