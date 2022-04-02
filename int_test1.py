# pip install pandas,numpy,matplotlib,sklearn,scipy,xlsxwriter,xlrd
# from logit import lgt
import pandas as pd
from draw import draw_bar


df1 = pd.read_csv(".\sample_train.csv", encoding='GBK')
df2 = pd.read_csv("sample_test.csv", encoding='GBK')
df3 = pd.read_csv("sample_oot.csv", encoding='GBK')

def gg():
    return 1



# M-num goto project/Structure/Bookmark/etc.

# C-` quick switch theme
# S-Esc hide
# duplicate line
# S-Delete cut-line

# C-4: copy paths
# C-1: Toggle Sticky Selection
# S-Enter: start new line
# C-M-Enter: start new line above
# C-+: extend selection

# S-up: move line up
# S-down: move line down

## search
# S-S navigate anything
# C-s search in current file
# C-s: Then C-S-r: replace
# C-c C-f find
# C-c C-t replace
# M-x toggle-regexp
# C-x b: switcher [switch buffer]
# M-F7: find usage

## recent
# C-c r:recent file
# C-c C-r:recent changes

# code refactor/generation/extract
# M-Enter:show fix
# C-x C-r: reformat code
# C-Enter: generate, for mare in code-tab
# C-c 2: inline
# C-c 3: extract, include method, variable, constant, field...
# C-c 4: rename


## tab
# C-c w:close tab
# M-Right:move to next tab
# M-Left:move to previous tab

## main
# C-z: undo
# C-S-z: redo
# C-x C-f: open
# C-0: increse font size
# C-9: decrese font size

# C-d duplicated line
# C-w C-S-w
# S-enter
# C-M-enter
# C-S-enter

## snippet
# main

## test
# C-c t: go to test; use unittest

## f-string
# type f"", then type in word between colon, and you will find some magic