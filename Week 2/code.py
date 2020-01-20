'''
-------------------------------------
 Assignment 2 - EE2703 (Jan-May 2020)
 Done by Akilesh Kannan (EE18B122)
 Created on 18/01/20
 Last Modified on 18/01/20
-------------------------------------
'''

# importing necessary libraries
import sys
import math
import cmath
import numpy as np
import pandas as pd

# To improve readability
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
RESISTOR = "R"
CAPACITOR = "C"
INDUCTOR = "L"
IVS = "V"
ICS = "I"
VCVS = "E"
VCCS = "G"
CCVS = "H"
CCCS = "F"

# Classes for each circuit component
class resistor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2

class inductor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2

class capacitor:
    def __init__(self, name, n1, n2, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2

class voltageSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.phase = phase

class currentSource:
    def __init__(self, name, n1, n2, val, phase=0):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.phase = phase

class vcvs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class vccs:
    def __init__(self, name, n1, n2, n3, n4, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class ccvs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName

class cccs:
    def __init__(self, name, n1, n2, vName, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.vSource = vName

# Convert a number in engineer's format to math
def enggToMath(enggNumber):
    lenEnggNumber = len(enggNumber)
    if enggNumber[lenEnggNumber-1] == 'k':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*1000
    elif enggNumber[lenEnggNumber-1] == 'm':
        base = int(enggNumber[0:lenEnggNumber-1])
        return base*0.001
    else:
        return float(enggNumber)

# Extracting the tokens from a Line
def line2tokens(spiceLine):
    tokens = []
    allWords = spiceLine.split()

    if allWords[0][0] == RESISTOR or allWords[0][0] == CAPACITOR or allWords[0][0] == INDUCTOR:
        return [allWords[0], allWords[1], allWords[2], float(allWords[3])]
    elif allWords[0][0] == IVS or allWords[0][0] == ICS:
        if allWords[3] == "dc":
            return [allWords[0], allWords[1], allWords[2], float(allWords[4])]
        else:
            return [allWords[0], allWords[1], allWords[2], float(allWords[4]), float(allWords[5])]

# Extract the frequency and name of the AC component from a SPICE Line
def getComponentAndFrequency(listOfAllACLines):
    table = []
    for x in listOfAllACLines:
        [componentName, frequency] = x.split()
        table.append(componentName)
        table.append(float(frequency))
    return table

# Print the Circuit Definition in the required format
def printCktDefn(SPICELinesTokens):
    for x in SPICELinesTokens:
        for y in x:
            print(y, end=' ')
        print('')
    print('')
    return

if __name__ == "__main__":

    # checking number of command line arguments
    if len(sys.argv)!=2 :
        sys.exit("Invalid number of arguments!")
    else:
        try:
            circuitFile = sys.argv[1]
            circuitFreq = 1e-100
            # checking if given netlist file is of correct type
            if (not circuitFile.endswith(".netlist")):
                print("Wrong file type!")
            else:
                SPICELines = []
                with open (circuitFile, "r") as f:
                    for line in f.readlines():
                        SPICELines.append(line.split('#')[0].split('\n')[0])
                try:
                    # finding the location of the identifiers
                    identifier1 = SPICELines.index(CIRCUIT_START)
                    identifier2 = SPICELines.index(CIRCUIT_END)

                    SPICELinesActual = SPICELines[identifier1+1:identifier2]
                    SPICELinesTokens = [line2tokens(line) for line in SPICELinesActual]
                    SPICELinesACLines = [line.split('.ac ')[1] for line in SPICELines if ".ac" in line]

                    if SPICELinesACLines:
                        acComponents = getComponentAndFrequency(SPICELinesACLines)

                    circuitFreq = acComponents[1]

                    for x in SPICELinesTokens:
                        if x[0][0] == IVS or x[0][0] == ICS:
                            if x[0] in acComponents:
                                x.append(acComponents[acComponents.index(x[0])+1])

                    # Printing Circuit Definition in Reverse Order
                    print("\nThe Circuit Definition is:\n")
                    printCktDefn(SPICELinesTokens)
                except ValueError:
                    sys.exit("Netlist does not abide to given format!")
        except FileNotFoundError:
            sys.exit("Given file does not exist!")
