# -------------------------------------
# Assignment 1 - EE2703 (Jan-May 2020)
# Done by Akilesh Kannan (EE18B122)
# Created on 16/01/20
# Last Modified on 17/01/20
# -------------------------------------

# importing necessary libraries
import sys

# To improve readability
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
RESISTOR = "R"
CAPACITOR = "C"
INDUCTOR = "L"

# Extracting the tokens from a Line
def line2tokens(spiceLine):
    tokens = []
    allWords = spiceLine.split()
    if(len(allWords)==4):
        elementName = allWords[0]
        node1 = allWords[1]
        node2 = allWords[2]
        value = allWords[3]
    return [elementName, node1, node2, value]

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
                        for x in SPICELinesTokens:
                            print(x)
                    except ValueError:
                        print("Netlist does not abide to given format!")
        except FileNotFoundError:
            print("Given file does not exist! Abort")
