import argparse
import platform, logging, os
import re
import csv
import ipaddress

#########Logging configuration##########
if platform.platform().startswith('Windows'):
    logging_file = os.path.join(os.getcwd(), os.path.splitext(os.path.basename(__file__))[0]+'.log')
else:
    logging_file = os.path.join(os.getcwd(), os.path.splitext(os.path.basename(__file__))[0]+'.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : %(levelname)s : %(threadName)-9s : %(message)s',
    filename=logging_file,
    filemode='w',
)
#########Logging configuration ends##########
__version__ = '1.0'

REG_EXPR = '^2016-0[78]-\d{2} \d{2}:\d{2}:\d{2},\d+\.\d+,\d+\.\d+\.\d+\.\d+,\d+\.\d+\.\d+\.\d+,\d+,\d+,[A-Z]+,[A-Z\.]{6},\d+,\d+,\d+,\d+,[a-z14]+[dst14]$'


def convert(line):
    outrow = []
    row = line.split(',')
    #print("type of row:",type(row),row)
    d, t = row[0].split(' ')
                # split the date: 2016-08-01 into 3 columns
    for val in d.split('-'):
        outrow.append(val)

                # split the time: 07:59:46 into 3 columns
    for val in t.split(':'):
        outrow.append(val)
    outrow.append(row[1])

                # convert IP adresses to integer value
    outrow.append(int(ipaddress.ip_address(row[2])))
    outrow.append(int(ipaddress.ip_address(row[3])))
    outrow.append(row[4])
    outrow.append(row[5])

                # convert: TCP -> 1 ; UDP -> 2 ; SCTP -> 3 & rest -> 4
    if row[6] == 'TCP':
        outrow.append(1)
    elif row[6] == 'UDP':
        outrow.append(2)
    elif row[6] == 'SCTP':
        outrow.append(3)
    else:
        outrow.append(4)

                # convert Flags to ASCII value & split it into different columns
    for ch in row[7]:
        outrow.append(ord(ch))

    outrow.append(row[8])
    outrow.append(row[9])
    outrow.append(row[10])
    outrow.append(row[11])

                # convert dos -> 0 & rest -> 1
    if row[12] == 'background\n':
        outrow.append(1)
    else:
        outrow.append(0)

    return outrow


def main(infile, outfile):

    if outfile == "":
        outfile = '/depot/datalab/sakhala/data/clean_data/output1.csv'
    with open(infile) as f:
        with open(outfile, 'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            r = re.compile(REG_EXPR)

            line = f.readline()
            while line:
                if r.match(line):
                    writer.writerow(convert(line))
                else:
                    pass

                line = f.readline()

if __name__ == "__main__":
    """
    Execution starts here.
    """

    parser = argparse.ArgumentParser(description='Run data cleaner script.')
    parser.add_argument('-i', '--input', help='Input filename', type=str, required=True)
    parser.add_argument('-o', '--out', help='Output file', type=str, default="")
    args = parser.parse_args()

    main(args.input, args.out)


    
