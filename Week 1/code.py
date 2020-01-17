# -------------------------------------
# Assignment 1 - EE2703 (Jan-May 2020)
# Done by Akilesh Kannan (EE18B122)
# Created on 16/01/20
# Version v0
# -------------------------------------

# importing necessary libraries
import sys

# To improve readability
CIRCUIT_START = ".circuit"
CIRCUIT_END = ".end"
RESISTOR = "R"
CAPACITOR = "C"
INDUCTOR = "L"

if __name__ == "__main__":
    # checking number of arguments
    if len(sys.argv)!=2 :
        print("Invalid number of arguments!")
    else:
        try:
            circuitFile = sys.argv[1]
            # checking if given netlist file is of correct type
            if (not circuitFile.endswith(".netlist")):
                print("Wrong file type!")
            else:
                with open (f, "r") as circuitFile:
                    SPICELines = []
                    [SPICELines.append(line.split('#')[0].split()) for line in f.readlines()]

        except FileNotFoundError:
            print("Given file does not exist! Abort")
