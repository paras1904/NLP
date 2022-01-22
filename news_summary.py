# -*- coding: utf-8 -*-
"""news_summary.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PsmwiJGi4gMeqrDhLFWrc5lqjpgwo78f
"""

!pip uninstall opencv-python-headless==4.5.5.62 
!pip install opencv-python-headless==4.1.2.30
!pip3 install easyocr
import easyocr

reader = easyocr.Reader(['en'])
output = reader.readtext('download.jpg')
text = [output[i][1] for i in range(len(output))]

text = [output[i][1] for i in range(len(output))]

!pip3 install transformers

def summery(s):
  from transformers import pipeline
  summr = pipeline("summarization")
  min_len = int(len(s.split())/3)
  max_len = int(len(s.split())/2)
  summari = summr(s,min_length=min_len,max_length=max_len)
  summari = [i.values() for i in summari]
  return str(summari)


