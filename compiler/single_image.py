#!/usr/bin/env python
import sys
import os
from argparse import ArgumentParser

def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--img_path', type=str,
                      dest='img_path', help='image filepath',
                      required=True)
  parser.add_argument('--output_path', type=str,
                      dest='output_path', help='output path directory',
                      required=True)
  parser.add_argument('--model_json_file', type=str,
                      dest='model_json_file', help='trained model json file',
                      required=True)
  parser.add_argument('--model_weights_file', type=str,
                      dest='model_weights_file', help='trained model weights file', required=True)

  return parser

def main():
    parser = build_parser()

if __name__ == "__main__":
  main()