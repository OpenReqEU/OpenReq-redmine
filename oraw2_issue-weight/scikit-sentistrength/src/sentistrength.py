import sys
import subprocess

import numpy as np

def getSentiment(text):
    try:
        out = subprocess.run(
            ['java', '-jar', '../sentiStrengthCom.jar', 'sentidata', '../data/french/', 'text', text.replace(' ', '+').encode('utf-8'), 'sentenceCombineTot', 'paragraphCombineTot', 'noDictionary', 'noDeleteExtraDuplicateLetters', 'alwaysSplitWordsAtApostrophes'],
            stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
        
        sent_pos, sent_neg = out.split(' ')
        return int(sent_pos), int(sent_neg)
    except:
        print(text, "TEXT E") # look at ARG_MAX errors
        return 0, 0

if __name__ == '__main__':
    print(getSentiment(sys.argv[1]))