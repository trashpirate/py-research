import csv
import numpy as np

def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b

def read_csv(filename):

    with open(filename,newline='') as csvfile:
        reader = csv.reader(csvfile)
        n = []
        for row in reader:
            m = []
            for col in row[:-1]:
                m.append(float(col))
            n.append(m)
        

    return np.array(n)

def read_csv_raw(filename):
    file = csv.reader(open(filename, 'r'))
    n = []
    for row in file:
        m = []
        for col in row:
            m.append(float(col))
        n.append(m)

    return np.array(n)

def read_csv_string(filename):
    file = csv.reader(open(filename, 'r'))
    n = []
    
    for row in file:
        m=[]
        for col in row:
            if col:
                if(isint(col)):
                    m.append(float(col))
                elif(isfloat(col)):
                    m.append(float(col))
                else:
                    m.append(col)
        n.append(m)

    return n

def read_txt(filename):

    file1 = open(filename, 'r') 
    Lines = file1.readlines() 
    
    line_array = [float(line) for line in Lines]
    return np.array(line_array)