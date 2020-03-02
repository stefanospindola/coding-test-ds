# Coding Challenge - Data Scientist

## The problem - Consultation finishing time estimation at a clinic

Here there is a model to predict the time a patient will leave a clinic, after having a \
consultation with a doctor, in an appointment-free service.

The clinic receives patients from 14:00 (time 0) to 18:00. Patients that arrived before 18:00 wait inside the clinic \
for their consultation.

There is a triage room with capacity for one patient at a time. As soon as it is available, arriving patients will pass \
a quick assessment to flag the patients that should have priority in seeing a doctor. At the end of this assessment, \
patients should be given an estimate of the time they will be free, i.e., the expected time for finishing their \
consultation. This repository have a system to give that estimate.

After the assessment, patients will wait until any of the 6 doctors working at the clinic are free. If none is free, \
patients will wait in a FIFO virtual queue. If a a new patient receives an `urgent` assessment, she will wait in the \
`urgent` FIFO virtual queue. Vacant doctors will always first receive patients from that queue. At the moment they call \
a new patient, doctors can see the size of both queues.

Consultations don't have a fixed duration. Instead, doctors are supposed to give patients the attention demanded by their \
cases.

The data provided refers to a continuous sequence of working days. The model is evaluated against the following 5 \
days.

### Procedure

1. Start a git clone repository with ```git clone https://github.com/stefanospindola/coding-test-ds.git```

### Training tool usage

1. Install Python 3 and cd to the solution folder
1. Run ```virtualenv env```
1. Run ```. env/bin/activate``` (commands might differ depending on your OS)
1. Run ```pip3 install -r requirements.txt```
1. Run ```python3 train.py data/```

### Evaluation tool usage

1. Install Python 3 and cd to the solution folder
1. Run ```virtualenv env```
1. Run ```. env/bin/activate``` (commands might differ depending on your OS)
1. Run ```pip3 install -r requirements.txt```
1. Run ```python3 evaluate.py data_test/```

## Presentation

https://docs.google.com/presentation/d/1ES9CpPzvu8dh8BQS_uOh8EMGcyRN-s_u-VcHCC4bukk/edit?usp=sharing




