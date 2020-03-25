#convert json to csv and then to arrow structure
import json
import csv as cv

from pyarrow import csv
import pandas as pd

new_dic ={}


with open('my_data.json') as json_file:
    data = json.load(json_file)

print(data)


def convertCSV(data,previous_char):
    for x,y in data.items():
        if(type(y)) == list or (type(y)) == dict:
            if previous_char =="":
                previous_char += str(x)+"_"
            else:
                previous_char += "_"+str(x)
        else:
            if previous_char =="":       
                new_dic[str(x)] = y     
            else:
                if previous_char[-1]!= '_':
                    previous_char += "_"+str(x)
                else:
                    previous_char += str(x)
                new_dic[previous_char] = y
                      
        if (type(y)) == dict:
            convertCSV(y, previous_char)
        elif (type(y)) == list:
            
            def convertList(list1,count,previous_char):
                for item in list1:
                    if (type(item)) != list and (type(item)) != dict:
                        previous_char += "_"+str(count)
                        count +=1
                        new_dic[previous_char] = item
                    elif (type(item)) == dict:
                        previous_char += "_"+str(count)
                        convertCSV(item, previous_char)
                        count+=1
                    else:
                        previous_char += "_"+str(count)
                        convertList(item, 0)
                    
                
                    previous_char = previous_char[:-2]
            count =0
            convertList(y,count,previous_char)
            
        previous_char = previous_char[:-2]     
        

convertCSV(data,"")

print(new_dic)


data_file = open('data_file.csv', 'w')
csv_writer = cv.writer(data_file)
header = new_dic.keys()
print(header)
csv_writer.writerow(header)        
csv_writer.writerow(new_dic.values())
data_file.close()

table = csv.read_csv('data_file.csv')
print(table)
df = table.to_pandas()
print(df)

