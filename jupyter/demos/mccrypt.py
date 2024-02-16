## Demo decryption of a substitution cipher by the Markov Chain Monte Carlo method.
## Intended to be used with a Jupyter notebook (markov-chain-demo.ipynb)
## Copyright (C) 2024 Chong Yidong

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## Default ciphertext file.
cipher_file = 'ciphertext.txt'

## A long source text, for generating the Markov chain.
source_file = 'lesmiserables.txt'

## The name of the file to which we write the Markov chain matrix.
weight_file = 'english'

import numpy as np
import random
import os.path
import sys

## Return a character corresponding to character code N (an integer).
## Character codes 0-25 map to A-Z, and 26 maps to SPC.
def mychr(n):
    if n == 26: return '_'
    return chr(n + 97) # code 97 is `a' in ASCII

## Return a text string corresponding to an array of character codes.
def text_to_string(int_array):
    return ''.join(map(mychr, int_array.tolist()))

## Return an array containing shuffled character codes 0-26.
def random_key():
    char_codes = list(range(27))
    random.shuffle(char_codes)
    return np.array(char_codes)

## Read a sample text file named INFILE, and construct a 2D array W,
## such that W[a,b] is the probability of character code b following
## character code a.  Save the resulting matrix to OUTFILE.
def build_matrix(infile, outfile):
    print('Building weight matrix...')
    W = zeros((27,27), dtype=uint64)
    lastn, nelts = -1, 0
    with open(infile) as f:
        while True:
            c = f.read(1)
            if not c: break
            n = ord(c)
            ## Downcase
            if n > 90: n -= 32
            ## Convert ASCII characters to character codes 0-25.
            ## Convert non-ASCII characters to n=26 (character
            ## separators).  Ignore multiple runs of such chars.
            if n < 65 or n > 90:
                if lastn == 26: continue
                n = 26
            else:
                n = n - 65
            ## Record this sample in the weights matrix.
            if lastn >= 0:
                W[lastn, n] += 1
                nelts += 1
            lastn = n
    print('Processed ' + str(nelts) + ' character pairs')
    ## Calculate a 2D array of probability weights
    W2 = zeros((27,27), dtype=float)
    for j in range(27):
        Wrow = np.copy(W[j,:]).astype(float)
        ## Avoid putting zeros in the matrix; apply some small
        ## positive number instead.
        for k in range(27):
            if W[j,k] == 0:
                Wrow[k] = 1.0e-6
        rowsum = sum(Wrow)
        W2[j,:] = Wrow / sum(Wrow)
    save(outfile, W2)

## Read the text in INFILE, and return its contents as a list of
## character codes.
def read_filechars(infile):
    charlist = []
    lastn = 26
    with open(infile) as f:
        while True:
            c = f.read(1)
            if not c: break
            n = ord(c)
            ## Downcase
            if n > 90:
                n -= 32
            if n < 65 or n > 90:
                ## Convert non-ASCII characters to n=26 (character
                ## separators).  Ignore multiple runs of such chars.
                if lastn == 26:
                    continue
                lastn = 26
            else:
                lastn = n - 65
            charlist.append(lastn)
    return charlist

def string_to_chars(s):
    charlist = []
    lastn = 26
    for k in range(len(s)):
        n = ord(s[k])
        ## Downcase
        if n > 90:
            n -= 32
        if n < 65 or n > 90:
            ## Convert non-ASCII characters to n=26 (character
            ## separators).  Ignore multiple runs of such chars.
            if lastn == 26:
                continue
            lastn = 26
        else:
            lastn = n - 65
        charlist.append(lastn)
    return charlist

## Given KEY, an array of character codes, swap 2-6 of its elements
## and return a copy.
def key_fluctuate(key):
    new_key = np.copy(key)
    ## Choose 2-6 characters to permute.
    nchoices = random.randint(2,6)
    indices = random.sample(range(27), nchoices)
    subkey  = key[indices]
    random.shuffle(subkey)
    new_key[indices] = subkey
    return new_key

## The key weight is log { prod_n W[p(n),p(n+1)] }
##                  == sum_n log { W[p(n),p(n+1)] }
def key_weight(key, ciphertext, weights):
    p = key[ciphertext] # Proposed plaintext
    return np.sum(np.log(weights[p[0:-1], p[1:]]))

## The Monte Carlo loop.
def mc_loop(N, ciphertext, weights, key):
    weight = key_weight(key, ciphertext, weights)
    for j in range(N):
        ## Generate a new move and calculate its weight.
        nkey    = key_fluctuate(key)
        nweight = key_weight(nkey, ciphertext, weights)
        allow_flip = True
        if nweight < weight:
            ## If the new weight is smaller than the old one, do the
            ## Metropolis-Hastings move: accept the move with
            ## probability prod M' / prod M.  The key weight
            ## calculated earlier is log(prod M).
            p = np.exp(nweight - weight)
            if random.random() > p:
                allow_flip = False
        if allow_flip:
            key = nkey
            weight = nweight
        # if j % 1000 == 0:
        #     print('Run ' + str(j) + ', Weight = ' + str(weight))
        #     print(text_to_string(key[ciphertext]))
        #     print('')
    return key

## Run the Monte Carlo simulation and print the results after every
## 1000 steps.
def main():
    ## Ciphertext file
    if len(sys.argv) > 1:
        f = sys.argv[1]
    else:
        f = cipher_file

    ## Regenerate the weight file if necessary.
    if not os.path.isfile(weight_file + '.npy'):
        build_matrix(source_file, weight_file)
    weights = np.load(weight_file + '.npy')

    ## Read the ciphertext, then run the MC loop.
    ciphertext = read_filechars(f)
    key = random_key()
    key = mc_loop(50000, ciphertext, weights, key)

#### DEMO using Jupyter Lab Widgets ######

## Global variables (saved across Jupyter notebook cells)

def encipher(plaintext):
    crypto_key = random_key()
    return crypto_key[plaintext]

demo_weights = []
demo_ciphertext = []

def run_demo():
    import ipywidgets as wg
    from IPython.display import display
    msg_button = wg.Button(description="Create Message")
    mc_button = wg.Button(description="Run Monte Carlo", disabled=True)
    output = wg.Output()

    s = 'Insert your desired text message here, then click on the "Create Message" button.  This generates a random caesar cipher and encrypts your text.  Capitalization is ignored, and all punctuation characters are treated as spaces.  Next, we will use Markov Chain Monte Carlo to decipher the encrypted message.  Note that longer messages are more likely to be correctly decrypted.'

    textarea = wg.Textarea(value=s, layout=wg.Layout(height="8em", width="auto"))
    display(textarea, msg_button, output, mc_button)

    global demo_key, demo_n, demo_ciphertext, demo_weights
    demo_key = []
    demo_n = 0
    demo_ciphertext = False
    if len(demo_weights) == 0:  # Reload the weights if necessary.
        demo_weights = np.load(weight_file + '.npy')

    def init_mc(b):
        global demo_key, demo_n, demo_ciphertext
        key = random_key()
        demo_n = 0
        demo_key = []
        demo_ciphertext = key[string_to_chars(textarea.value)]
        output.clear_output()
        with output:
            print("Cipher:")
            print("abcdefghijklmnopqrstuvwxyz_")
            print(text_to_string(key))
            print("")
            print("Encrypted Message (message length: {}):".format(len(demo_ciphertext)))
            print(text_to_string(demo_ciphertext))
        mc_button.disabled = False

    def run_mc(b):
        global demo_key, demo_ciphertext, demo_weights, demo_n
        nsteps = 20000
        with output:
            if len(demo_ciphertext) == 0:
                print("No ciphertext intialized")
            else:
                if len(demo_key) == 0:
                    demo_n = 0
                    demo_key = random_key()
                demo_key = mc_loop(nsteps, demo_ciphertext, demo_weights, demo_key)
                demo_n += nsteps
                print("")
                print("After {} steps:".format(demo_n))
                print(text_to_string(demo_key[demo_ciphertext]))

    msg_button.on_click(init_mc)
    mc_button.on_click(run_mc)

def run_energy_demo():
    import ipywidgets as wg
    from IPython.display import display
    msg_button = wg.Button(description="Create Message")
    mc_button = wg.Button(description="Run Monte Carlo", disabled=True)
    output = wg.Output()

    global demo_weights
    if len(demo_weights) == 0:
        demo_weights = np.load(weight_file + '.npy')

    def string_energy(string):
        p = string_to_chars(string)
        return - np.sum(np.log(demo_weights[p[0:-1], p[1:]]))

    t1 = wg.Text(value="java", layout=wg.Layout(width="6em"))
    t2 = wg.Text(value="jtva", layout=wg.Layout(width="6em"))
    E1 = wg.Label(value=" energy = {}".format(string_energy(t1.value)))
    E2 = wg.Label(value=" energy = {}".format(string_energy(t2.value)))
    display(s, wg.HBox([t1, E1]), wg.HBox([t2, E2]))

    def on_change_1(change):
        if change["new"] != change["old"]:
            E1.value = " energy = {}".format(string_energy(change["new"]))
    t1.observe(on_change_1, names =["value"])

    def on_change_2(change):
        if change["new"] != change["old"]:
            E2.value = " energy = {}".format(string_energy(change["new"]))
    t2.observe(on_change_2, names =["value"])
