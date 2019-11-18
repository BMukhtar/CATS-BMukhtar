import os
import pathlib
import shutil
import subprocess
import sys
from os.path import abspath, dirname, exists, join

try:
    import segment
    import segment_input_text

except:
    _this_dir = dirname(abspath(__file__))
    sys.path.insert(0, join(_this_dir, '..'))
    import segment
    import segment_input_text


def tearDown():
    test_output_path = "/tmp/aganesh_cats"
    if exists(test_output_path):
        shutil.rmtree(test_output_path)


def test_with_input_dir():
    test_output_path = "/tmp/aganesh_cats"
    if not os.path.isdir(test_output_path):
        try:
            os.makedirs(test_output_path, exist_ok=True)
        except Exception as e:
            print(e)

    segment.main(join(_this_dir, "input"), test_output_path)

    orig_output = join(_this_dir, "output", "wiki_sample.txt.seg")

    test_output = os.path.join(test_output_path, "wiki_sample.txt.seg")

    with open(orig_output) as orig_file, open(test_output) as test_file:
      for orig_line, test_line in zip(orig_file, test_file):
        orig_score = orig_line.split()[-1]
        test_score = orig_line.split()[-1]
        assert orig_score[:5] == test_score[:5]
  #  assert 1 == 0


def test_with_input_file():
    input_text = open(join(_this_dir, 'input', 'wiki_sample.txt')).read()
    segmented_text = segment_input_text.main(input_text)

    orig_output = join(_this_dir, "output", "wiki_sample.txt.seg")

    orig_lines = open(orig_output).readlines()
    for orig_line, test_line in zip(orig_lines, segmented_text):
        orig_score = orig_line.split()[-1]
        test_score = orig_line.split()[-1]
        assert orig_score[:5] == test_score[:5]
