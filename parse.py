import csv
key = '15'
outputName = './output/'+key+'_output.csv'
result = []
source = ['2015 NAV.csv', '2016 NAV.csv', '2017 NAV.csv',
          '201711-12 NAV.csv', '2018 NAV.csv', '2019 NAV.csv']

lastPrice = 0
for fileName in source:
    with open(fileName, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[1] == key:
                if row[2] != 'NULL':
                    lastPrice = float('{:.3f}'.format(float(row[2])))
                temp = []
                temp.append(row[0].split(' ')[0])
                temp.append(lastPrice)
                temp.append(lastPrice)
                temp.append(lastPrice)
                temp.append(lastPrice)
                temp.append(0)
                result.append(temp)

with open(outputName, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    for item in result:
        writer.writerow(item)
