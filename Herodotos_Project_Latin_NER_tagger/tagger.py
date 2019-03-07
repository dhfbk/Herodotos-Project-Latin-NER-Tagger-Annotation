#!/usr/bin/env python
from __future__ import print_function

import random
import os
import re
import sys
import time
import codecs
import optparse
import json
import numpy as np
import scipy.io
# from loader import prepare_sentence
# from utils import create_input, iobes_iob, iob_ranges, zero_digits
# from model import Model
import fileinput
import theano
import theano.tensor as T
import pickle
### defaults from utils.py
# models_path = "./models"
# eval_path = "./evaluation"
# eval_temp = os.path.join(eval_path, "temp")
# eval_script = os.path.join(eval_path, "conlleval")

from tagger_utils import *

floatX = theano.config.floatX
device = theano.config.device


######################################################################

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-i", "--input", default=None, # sys.argv[1],
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default=sys.stdout,
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
optparser.add_option(
    "-M", "--model_loc", default="Mila_Kunis",
    help="Name of the relevant model within ./models/"
)
optparser.add_option(
    "--outputFormat", default="pelagios",
    help="Output file format"
)
optparser.add_option(
    "--inputFormat", default="tok",
    help="Output file format"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter

# Load existing model
sys.stderr.write("Loading model...")
### HARD CODED MODEL LOCATION FOR THE PACKAGED VERSION
opts.model_loc = 'latin_ner'
workingDir = os.path.dirname(os.path.realpath(sys.argv[0]))
opts.model = os.path.join(workingDir, opts.model_loc)
###

# Check parameters validity
assert os.path.isdir(opts.model)
# assert os.path.isfile(opts.input) or None

model = Model(opts.model_loc, model_path=opts.model)
parameters = model.parameters

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval = model.build(training=False, **parameters)
model.reload()



#############################
sys.stderr.write('Tagging...')

if opts.inputFormat == 'tok':

    # f_output = open(opts.output, 'w')
    words = []
    # INPUT = (open(opts.input).read().splitlines())
    ### CONVERT STD INPUT TO LIST OF WORDS
    # STANDARD INPUT SHOULD BE ONE TOKENIZED SENTENCE PER LINE
        # ADD FUNCTIONALITY TO TOKENIZE ON THE FLY!!!
    # NEED TO CONVERT THAT TO ONE TOKEN PER LINE WITH BLANK LINE FOR SPACE
    temp_f = '{}/temp.{}.txt'.format('/'.join(opts.model.split('/')), str(random.randint(0,1000000000000000000000)))
    f_temp = open(temp_f, 'w')
    for line in sys.stdin:
        for word in line.split():
            f_temp.write('{} 0\n'.format(word))
        f_temp.write('\n')
    f_temp.close()
    ### PROCESS INPUT

    INPUT = (open(temp_f).read().splitlines())
    endLine = len(INPUT)
    ind = 0
    for line in INPUT:
        ind += 1
        count = 0

        # Lowercase sentence
        if parameters['lower']:
            line = line.lower()
        # Replace all digits with zeros
        if parameters['zeros']:
            line = zero_digits(line)

        line = line.split()
        if len(line) > 0:
            words.append(line[0])

        if len(line) == 0 or ind == endLine:

            if len(words) > 0:
                # Prepare input
                sentence = prepare_sentence(words, word_to_id, char_to_id,
                                            lower=parameters['lower'])
                input = create_input(sentence, parameters, False)
                # Decoding
                if parameters['crf']:
                    try:
                        y_preds = np.array(f_eval(*input))[1:-1]
                    except:
                        y_preds = np.array([0] * len(words))
                else:
                    try:
                        y_preds = f_eval(*input).argmax(axis=1)
                    except:
                        y_preds = np.array([0] * len(words))
                y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                # Output tags in the IOB2 format
                if parameters['tag_scheme'] == 'iobes':
                    y_preds = iobes_iob(y_preds)
                # Write tags
                assert len(y_preds) == len(words)
                
                if opts.outputFormat == 'json':
                    sys.stdout.write(json.dumps({ "text": ' '.join(words), "ranges": iob_ranges(y_preds) }))

                elif opts.outputFormat == 'pelagios':

                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    sys.stdout.write('{}\n'.format(', '.join(str(tup) for tup in tuples)).replace("'",'').replace('[','(').replace(']',')'))

                elif opts.outputFormat == 'crf':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        try:
                            sys.stdout.write('{}\t{}\n'.format(y, w))
                        except:
                            sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))
                    sys.stdout.write('\n')


                elif opts.outputFormat == 'conll':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                            sys.stdout.write('{}\t{}\n'.format(w, y))
                        except:
                            sys.stdout.write('UNICODE-ERROR\t{}\n'.format(y))
                    sys.stdout.write('\n')


                elif opts.outputFormat == 'list':


                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    for tup in tuples:
                        ne = tup[1]
                        label = tup[2]

                        if label not in uniqueNEs:
                            uniqueNEs[label] = {}
                        if ne not in uniqueNEs[label]:
                            uniqueNEs[label][ne] = 0
                        uniqueNEs[label][ne] += 1

                else:
                    print('UNSUPPORTED OUTPUT FORMAT {}'.format(opts.outputFormat))

                words = []

    if opts.outputFormat == 'list':

        for label in uniqueNEs:
            print(label)

            orderedNEs = []
            for ne in uniqueNEs[label]:
                orderedNEs.append([uniqueNEs[label][ne], ne])
            orderedNEs.sort(reverse=True)
            for tup in orderedNEs:
                ne = tup[1]
                count = str(tup[0])
                print('\t{}\t{}'.format(ne, count))
            print()

    os.system('rm '+temp_f)
    # sys.stderr.close()

elif opts.inputFormat == 'conll':

    uniqueNEs = {}
    words = []

    temp_f = '{}/temp.{}.txt'.format('/'.join(opts.model.split('/')), str(random.randint(0,1000000000000000000000)))
    f_temp = open(temp_f, 'w')
    for line in sys.stdin:
        line = line.replace('\n','').replace('\r','')
        f_temp.write('{}\n'.format(line))
    f_temp.close()
    ### PROCESS INPUT


    INPUT = (open(temp_f).read().splitlines())
    endLine = len(INPUT)
    ind = 0
    for line in INPUT:
        ind += 1
        count = 0

        # Lowercase sentence
        if parameters['lower']:
            line = line.lower()
        # Replace all digits with zeros
        if parameters['zeros']:
            line = zero_digits(line)

        line = line.split()
        if len(line) > 0:
            words.append(line[0])

        if len(line) == 0 or ind == endLine:

            if len(words) > 0:
                # Prepare input
                sentence = prepare_sentence(words, word_to_id, char_to_id,
                                            lower=parameters['lower'])
                input = create_input(sentence, parameters, False)
                # Decoding
                if parameters['crf']:
                    try:
                        y_preds = np.array(f_eval(*input))[1:-1]
                    except:
                        y_preds = np.array([0] * len(words))
                else:
                    try:
                        y_preds = f_eval(*input).argmax(axis=1)
                    except:
                        y_preds = np.array([0] * len(words))
                y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                # Output tags in the IOB2 format
                if parameters['tag_scheme'] == 'iobes':
                    y_preds = iobes_iob(y_preds)
                # Write tags
                assert len(y_preds) == len(words)
                
                if opts.outputFormat == 'json':
                    sys.stdout.write(json.dumps({ "text": ' '.join(words), "ranges": iob_ranges(y_preds) }))

                elif opts.outputFormat == 'pelagios':

                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    sys.stdout.write('{}\n'.format(', '.join(str(tup) for tup in tuples)).replace("'",'').replace('[','(').replace(']',')'))

                elif opts.outputFormat == 'crf':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        try:
                            sys.stdout.write('{}\t{}\n'.format(y, w))
                        except:
                            sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))
                    sys.stdout.write('\n')


                elif opts.outputFormat == 'conll':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                            sys.stdout.write('{}\t{}\n'.format(w, y))
                        except:
                            sys.stdout.write('UNICODE-ERROR\t{}\n'.format(y))
                    sys.stdout.write('\n')


                elif opts.outputFormat == 'list':


                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    for tup in tuples:
                        ne = tup[1]
                        label = tup[2]

                        if label not in uniqueNEs:
                            uniqueNEs[label] = {}
                        if ne not in uniqueNEs[label]:
                            uniqueNEs[label][ne] = 0
                        uniqueNEs[label][ne] += 1

                else:
                    print('UNSUPPORTED OUTPUT FORMAT {}'.format(opts.outputFormat))

                words = []

    if opts.outputFormat == 'list':

        for label in uniqueNEs:
            print(label)

            orderedNEs = []
            for ne in uniqueNEs[label]:
                orderedNEs.append([uniqueNEs[label][ne], ne])
            orderedNEs.sort(reverse=True)
            for tup in orderedNEs:
                ne = tup[1]
                count = str(tup[0])
                print('\t{}\t{}'.format(ne, count))
            print()

    os.system('rm '+temp_f)







elif opts.inputFormat == 'crf':

    uniqueNEs = {}
    words = []

    temp_f = '{}/temp.{}.txt'.format('/'.join(opts.model.split('/')), str(random.randint(0,1000000000000000000000)))
    f_temp = open(temp_f, 'w')
    for line in sys.stdin:
        line = line.split()
        if len(line) == 0:
            f_temp.write('\n')
        else:
            f_temp.write('{}\tO\n'.format(line[1]))
    f_temp.close()
    ### PROCESS INPUT


    INPUT = (open(temp_f).read().splitlines())
    endLine = len(INPUT)
    ind = 0
    for line in INPUT:
        ind += 1
        count = 0

        # Lowercase sentence
        if parameters['lower']:
            line = line.lower()
        # Replace all digits with zeros
        if parameters['zeros']:
            line = zero_digits(line)

        line = line.split()
        if len(line) > 0:
            words.append(line[0])

        if len(line) == 0 or ind == endLine:

            if len(words) > 0:
                # Prepare input
                sentence = prepare_sentence(words, word_to_id, char_to_id,
                                            lower=parameters['lower'])
                input = create_input(sentence, parameters, False)
                # Decoding
                if parameters['crf']:
                    try:
                        y_preds = np.array(f_eval(*input))[1:-1]
                    except:
                        y_preds = np.array([0] * len(words))
                else:
                    try:
                        y_preds = f_eval(*input).argmax(axis=1)
                    except:
                        y_preds = np.array([0] * len(words))
                y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
                # Output tags in the IOB2 format
                if parameters['tag_scheme'] == 'iobes':
                    y_preds = iobes_iob(y_preds)
                # Write tags
                assert len(y_preds) == len(words)
                
                if opts.outputFormat == 'json':
                    sys.stdout.write(json.dumps({ "text": ' '.join(words), "ranges": iob_ranges(y_preds) }))

                elif opts.outputFormat == 'pelagios':

                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    sys.stdout.write('{}\n'.format(', '.join(str(tup) for tup in tuples)).replace("'",'').replace('[','(').replace(']',')'))

                elif opts.outputFormat == 'crf':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        try:
                            sys.stdout.write('{}\t{}\n'.format(y, w))
                        except:
                            sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))
                    sys.stdout.write('\n')

                elif opts.outputFormat == 'conll':

                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                            sys.stdout.write('{}\t{}\n'.format(w, y))
                        except:
                            sys.stdout.write('UNICODE-ERROR\t{}\n'.format(y))
                    sys.stdout.write('\n')

                elif opts.outputFormat == 'list':


                    offset = 0
                    tuples = []
                    for n in range(len(words)):
                        w = words[n]
                        y = y_preds[n]
                        if y == 'O':
                            y = '0'
                        else:
                            y = y.split('-')
                            y = '{}-{}'.format(y[1],y[0])
                        # try:
                            # sys.stdout.write('{}\t{}\n'.format(y, w))
                        # except:
                            # sys.stdout.write('{}\tUNICODE-ERROR\n'.format(y))


                        if y != '0':
                            if '-B' in y:
                                tuples.append([offset, w, y.split('-')[0]])
                            else:
                                tuples[-1][1] += ' {}'.format(w)

                        # sys.stdout.write('{} '.format())
                        offset += len(w) + 1

                    for tup in tuples:
                        ne = tup[1]
                        label = tup[2]

                        if label not in uniqueNEs:
                            uniqueNEs[label] = {}
                        if ne not in uniqueNEs[label]:
                            uniqueNEs[label][ne] = 0
                        uniqueNEs[label][ne] += 1

                else:
                    print('UNSUPPORTED OUTPUT FORMAT {}'.format(opts.outputFormat))

                words = []

    if opts.outputFormat == 'list':

        for label in uniqueNEs:
            print(label)

            orderedNEs = []
            for ne in uniqueNEs[label]:
                orderedNEs.append([uniqueNEs[label][ne], ne])
            orderedNEs.sort(reverse=True)
            for tup in orderedNEs:
                ne = tup[1]
                count = str(tup[0])
                print('\t{}\t{}'.format(ne, count))
            print()

    os.system('rm '+temp_f)

else:
    print('UNSUPPORTED INPUT FORMAT {}'.format(opts.inputFormat))
