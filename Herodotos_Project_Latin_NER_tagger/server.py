import time
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
import optparse
import sys
import os
from tagger_utils import *

HOST_NAME = 'localhost'
PORT_NUMBER = 9006

class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('content-length'))
        messageb = self.rfile.read(length)
        # print(messageb)
        message = json.loads(messageb.decode("UTF-8"))

        # print(message)
        outputFormat = message['outputFormat']
        out_str = ""

        uniqueNEs = {}
        words = []

        ind = 0
        for line in message['text'].splitlines():
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

            if len(line) == 0:

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
                    
                    if outputFormat == 'json':
                        out_str += json.dumps({ "text": ' '.join(words), "ranges": iob_ranges(y_preds) })

                    elif outputFormat == 'pelagios':

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

                            if y != '0':
                                if '-B' in y:
                                    tuples.append([offset, w, y.split('-')[0]])
                                else:
                                    tuples[-1][1] += ' {}'.format(w)

                            offset += len(w) + 1

                        out_str += '{}\n'.format(', '.join(str(tup) for tup in tuples)).replace("'",'').replace('[','(').replace(']',')')

                    elif outputFormat == 'crf':

                        for n in range(len(words)):
                            w = words[n]
                            y = y_preds[n]
                            if y == 'O':
                                y = '0'
                            else:
                                y = y.split('-')
                                y = '{}-{}'.format(y[1],y[0])
                            try:
                                out_str += '{}\t{}\n'.format(y, w)
                            except:
                                out_str += '{}\tUNICODE-ERROR\n'.format(y)
                        out_str += '\n'


                    elif outputFormat == 'conll':

                        for n in range(len(words)):
                            w = words[n]
                            y = y_preds[n]
                            try:
                                out_str += '{}\t{}\n'.format(w, y)
                            except:
                                out_str += 'UNICODE-ERROR\t{}\n'.format(y)
                        out_str += '\n'


                    elif outputFormat == 'list':


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

                            if y != '0':
                                if '-B' in y:
                                    tuples.append([offset, w, y.split('-')[0]])
                                else:
                                    tuples[-1][1] += ' {}'.format(w)

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
                        out_str += 'UNSUPPORTED OUTPUT FORMAT {}\n'.format(outputFormat)

                    words = []

        if outputFormat == 'list':

            for label in uniqueNEs:
                out_str += '{}\n'.label

                orderedNEs = []
                for ne in uniqueNEs[label]:
                    orderedNEs.append([uniqueNEs[label][ne], ne])
                orderedNEs.sort(reverse=True)
                for tup in orderedNEs:
                    ne = tup[1]
                    count = str(tup[0])
                    out_str += '\t{}\t{}\n'.format(ne, count)
                out_str += "\n"

        res = {}

        res["output"] = out_str

        self.send_response(200)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        ret = bytes(json.dumps(res), 'UTF-8')
        self.wfile.write(ret)


optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="",
    help="Model location"
)
optparser.add_option(
    "-M", "--model_loc", default="Mila_Kunis",
    help="Name of the relevant model within ./models/"
)
opts = optparser.parse_args()[0]

# Check parameters validity
# assert opts.delimiter

sys.stderr.write("Loading model...\n")
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
sys.stderr.write("Ready to tag!\n")

# MAIN

# simplifier_es = LexicalSimplifier("CASSAurusES.db", "stopwords-es.json")
# simplifier_gl = LexicalSimplifier("CASSAurusGAL.db", "stopwords-gl.json")
httpd = HTTPServer((HOST_NAME, PORT_NUMBER), MyHandler)
print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
httpd.server_close()
print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
