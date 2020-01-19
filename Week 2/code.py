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
    def __init__(self, name, val, n1, n2, n3, n4, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class vccs:
    def __init__(self, name, val, n1, n2, n3, n4, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2
        self.node3 = n3
        self.node4 = n4

class ccvs:
    def __init__(self, name, val, n1, n2, vName, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2

class cccs:
    def __init__(self, name, val, n1, n2, vName, val):
        self.name = name
        self.value = val
        self.node1 = n1
        self.node2 = n2

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

    # R, L, C, Independent Sources
    if(len(allWords) == 4):
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        value = allWords[3]
        return [elementName, node1, node2, enggToMath(value)]

    # CCxS
    elif(len(allWords) == 5):
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        voltageSource = allWords[3]
        value = allWords[4]
        return [elementName, node1, node2, voltageSource, enggToMath(value)]

    # VCxS
    elif(len(allWords) == 6):
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        voltageSourceNode1 = allWords[3]
        voltageSourceNode2 = allWords[4]
        value = allWords[5]
        return [elementName, node1, node2, voltageSourceNode1, voltageSourceNode2, enggToMath(value)]

    else:
        return []

# Extract the frequency and name of the AC component from a SPICE Line
def getComponentAndFrequency(listOfAllACLines):
    table = []
    for x in listOfAllACLines:
        [componentName, frequency] = x.split()
        table.append([componentName, float(frequency)])
    return table

# Print the Circuit Definition in the required format
def printCktDefn(SPICELinesTokens):
    for x in SPICELinesTokens[::-1]:
        for y in x[::-1]:
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

            # checking if given netlist file is of correct type
            if (not circuitFile.endswith(".netlist")):
                print("Wrong file type!")
            else:
                with open (circuitFile, "r") as f:
                    SPICELines = []
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

                        # Printing Circuit Definition in Reverse Order
                        print("\nThe Circuit Definition is:\n")
                        printCktDefn(SPICELinesTokens)
                    except ValueError:
                        sys.exit("Netlist does not abide to given format!")
        except FileNotFoundError:
            sys.exit("Given file does not exist!")
