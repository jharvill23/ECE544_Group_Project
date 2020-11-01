# ECE544_Group_Project
Estimating number of uniquely-perceivable voices

Step 1: Adjust root, text, and speaker_info parameters
in config.yml according to where you store VCTK on
disk.

Step 2: Run preprocessing.py with STAGE='012'. This
converts .flac files to .wav files (saves processing
time down the line) and removes silence from
beginning and end of recordings.

Step 3: Run extract_features.py

Step 4: Run continuous_spk_embed.py (extract speaker
embeddings and save to disk). Note, depending on
how much RAM you have, you may need to adjust
MAX_PROCESSES. With 8 cpus, it takes over 20GB
memory to load the speaker embedding model 8 separate
times for multiprocessing.
