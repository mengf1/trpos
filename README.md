# Welcome to Part-Of-Speech tagging for low-resource languages

## Introduction

This source code is the basis of the following paper:
> [Model Transfer for Tagging Low-resource Languages using a Bilingual Dictionary](http://www.aclweb.org/anthology/P/P17/P17-2093.pdf), by Meng Fang and Trevor Cohn, ACL 2017

## Building
It's developed on dynet toolkit.
- Install dynet following [clab/dynet](https://github.com/clab/dynet).
- Add the source code to folder *dynet/examples* and modify *CMakeLists.txt*.
- Make again.

## Code
- UniTagger: a universal POS tagger
- JointTagger: a tranfer learning model using both the gold and distant data

## Data format
The format of input data is as follows:
```
Tok_1 Tok_2 ||| Tag_1 Tag_2
Tok_1 Tok_2 Tok_3 ||| Tag_1 Tag_2 Tag_3
...
```

## Data resource

- [Crosslingual word embeddings and corpora](http://128.2.220.95/multilingual/data)
- [Universal POS tagset](https://github.com/slavpetrov/universal-pos-tags)
